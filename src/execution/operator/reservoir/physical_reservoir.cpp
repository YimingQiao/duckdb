#include "duckdb/execution/operator/reservoir/physical_reservoir.hpp"
#include "duckdb/parallel/task_scheduler.hpp"
#include "duckdb/storage/temporary_memory_manager.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/main/query_profiler.hpp"
#include "duckdb/parallel/thread_context.hpp"
#include "duckdb/storage/buffer_manager.hpp"

namespace duckdb {

PhysicalReservoir::PhysicalReservoir(LogicalOperator &op, vector<LogicalType> types, idx_t estimated_cardinality)
    : PhysicalOperator(PhysicalOperatorType::RESERVOIR, std::move(types), estimated_cardinality), is_impounding(true) {
}

//===--------------------------------------------------------------------===//
// Operator
//===--------------------------------------------------------------------===//
class ReservoirGlobalOperatorState : public GlobalOperatorState {
public:
	ReservoirGlobalOperatorState(const PhysicalReservoir &op_p, ClientContext &context_p)
	    : context(context_p), op(op_p),
	      num_threads(NumericCast<idx_t>(TaskScheduler::GetScheduler(context).NumberOfThreads())),
	      buffer_manager(BufferManager::GetBufferManager(context)) {
		global_buffer = make_uniq<ColumnDataCollection>(buffer_manager, op_p.types);
	}

public:
	ClientContext &context;
	const PhysicalReservoir &op;

	const idx_t num_threads;

	BufferManager &buffer_manager;

	//! Global Buffer
	unique_ptr<ColumnDataCollection> global_buffer;

	//! The number of active local states
	atomic<idx_t> active_local_states;

	//! Buffer held by each thread
	vector<unique_ptr<ColumnDataCollection>> local_buffers;

	//! Whether or not we have started scanning data using GetData
	atomic<bool> scanned_data;
};

class ReservoirOperatorState : public OperatorState {
public:
	ReservoirOperatorState(const PhysicalReservoir &op, ClientContext &context, ReservoirGlobalOperatorState &gstate)
	    : buffer_merged(false) {
		auto &buffer_manager = BufferManager::GetBufferManager(context);
		buffer = make_uniq<ColumnDataCollection>(buffer_manager, op.types);
		gstate.active_local_states++;
	}

public:
	//! Thread-local buffer
	unique_ptr<ColumnDataCollection> buffer;
	bool buffer_merged;
};

unique_ptr<OperatorState> PhysicalReservoir::GetOperatorState(ExecutionContext &context) const {
	auto &gstate = op_state->Cast<ReservoirGlobalOperatorState>();
	return make_uniq<ReservoirOperatorState>(*this, context.client, gstate);
}
unique_ptr<GlobalOperatorState> PhysicalReservoir::GetGlobalOperatorState(ClientContext &context) const {
	return make_uniq<ReservoirGlobalOperatorState>(*this, context);
}

OperatorResultType PhysicalReservoir::Execute(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
                                              GlobalOperatorState &gstate, OperatorState &state) const {
	auto &op_gstate = gstate.Cast<ReservoirGlobalOperatorState>();
	auto &op_state = state.Cast<ReservoirOperatorState>();

	if (is_impounding) {
		op_state.buffer->Append(input);
	} else {
		if (!op_state.buffer_merged) {
			std::cerr << "[Reservoir] Merge Buffer Finally\n";
			auto guard = op_gstate.Lock();
			op_gstate.local_buffers.push_back(std::move(op_state.buffer));

			// if all local buffer have been collected
			if (op_gstate.local_buffers.size() == op_gstate.active_local_states) {
				auto &global_buffer = *op_gstate.global_buffer;

				// combine local buffers
				for (auto &local_buf : op_gstate.local_buffers) {
					global_buffer.Combine(*local_buf);
				}
			}
			op_state.buffer_merged = true;
		}

		chunk.Reference(input);
	}
	return OperatorResultType::NEED_MORE_INPUT;
}

OperatorFinalizeResultType PhysicalReservoir::FinalExecute(ExecutionContext &context, DataChunk &chunk,
                                                           GlobalOperatorState &gstate, OperatorState &state_p) const {
	auto &op_gstate = gstate.Cast<ReservoirGlobalOperatorState>();
	auto &op_state = state_p.Cast<ReservoirOperatorState>();

	if (!op_state.buffer_merged) {
		auto guard = op_gstate.Lock();
		op_gstate.local_buffers.push_back(std::move(op_state.buffer));

		// if all local buffer have been collected
		if (op_gstate.local_buffers.size() == op_gstate.active_local_states) {
			auto &global_buffer = *op_gstate.global_buffer;

			// combine local buffers
			for (auto &local_buf : op_gstate.local_buffers) {
				global_buffer.Combine(*local_buf);
			}
		}
		op_state.buffer_merged = true;
	}

	chunk.SetCardinality(0);
	return OperatorFinalizeResultType::FINISHED;
}

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
enum class ReservoirSourceStage : uint8_t { INIT, SCAN_BUFFER, DONE };

class ReservoirLocalSourceState;

class ReservoirGlobalSourceState : public GlobalSourceState {
public:
	ReservoirGlobalSourceState(const PhysicalReservoir &op, const ClientContext &context);

	//! Initialize this source state using the info in the op_gstate
	void Initialize(ReservoirGlobalOperatorState &op_gstate);
	//! Prepare the scan_buf stage
	void PrepareScan(ReservoirGlobalOperatorState &op_gstate);
	//! Assigns a task to a local source state
	bool AssignTask(ReservoirGlobalOperatorState &gstate, ReservoirLocalSourceState &lstate);

	idx_t MaxThreads() override {
		D_ASSERT(op.op_state);
		auto &gstate = op.op_state->Cast<ReservoirGlobalOperatorState>();

		idx_t count = gstate.global_buffer->Count();
		return count / ((idx_t)STANDARD_VECTOR_SIZE * parallel_scan_chunk_count);
	}

public:
	const PhysicalReservoir &op;

	//! For synchronizing
	atomic<ReservoirSourceStage> global_stage;

	//! For buffer scan synchronization
	idx_t scan_chunk_idx = DConstants::INVALID_INDEX;
	idx_t scan_chunk_count;
	idx_t scan_chunk_done;
	idx_t scan_chunks_per_thread = DConstants::INVALID_INDEX;

	idx_t parallel_scan_chunk_count;
};

struct ReservoirScanState {
public:
	ReservoirScanState() : chunk_idx(DConstants::INVALID_INDEX) {
	}

	idx_t chunk_idx;

private:
	//! Implicit copying is not allowed
	ReservoirScanState(const ReservoirScanState &) = delete;
};

class ReservoirLocalSourceState : public LocalSourceState {
public:
	ReservoirLocalSourceState(const PhysicalReservoir &op, const ReservoirGlobalOperatorState &op_gstate,
	                          Allocator &allocator);

	//! Do the work this thread has been assigned
	void ExecuteTask(ReservoirGlobalOperatorState &op_gstate, ReservoirGlobalSourceState &gstate, DataChunk &chunk);
	//! Whether this thread has finished the work it has been assigned
	bool TaskFinished() const;
	//! Scan
	void ScanBuf(ReservoirGlobalOperatorState &op_gstate, ReservoirGlobalSourceState &gstate, DataChunk &chunk);

public:
	//! The stage that this thread was assigned work for
	ReservoirSourceStage local_stage;

	idx_t scan_chunk_idx_from = DConstants::INVALID_INDEX;
	idx_t scan_chunk_idx_to = DConstants::INVALID_INDEX;

	unique_ptr<ReservoirScanState> scan_state;
};

unique_ptr<GlobalSourceState> PhysicalReservoir::GetGlobalSourceState(ClientContext &context) const {
	return make_uniq<ReservoirGlobalSourceState>(*this, context);
}

unique_ptr<LocalSourceState> PhysicalReservoir::GetLocalSourceState(ExecutionContext &context,
                                                                    GlobalSourceState &gstate) const {
	return make_uniq<ReservoirLocalSourceState>(*this, op_state->Cast<ReservoirGlobalOperatorState>(),
	                                            BufferAllocator::Get(context.client));
}

ReservoirGlobalSourceState::ReservoirGlobalSourceState(const PhysicalReservoir &op, const ClientContext &context)
    : op(op), global_stage(ReservoirSourceStage::INIT), parallel_scan_chunk_count(1) {
}

void ReservoirGlobalSourceState::Initialize(ReservoirGlobalOperatorState &op_gstate) {
	auto guard = Lock();
	if (global_stage != ReservoirSourceStage::INIT) {
		// Another thread initialized
		return;
	}
	PrepareScan(op_gstate);
}

void ReservoirGlobalSourceState::PrepareScan(ReservoirGlobalOperatorState &op_gstate) {
	D_ASSERT(global_stage != ReservoirSourceStage::SCAN_BUFFER);
	auto &buf = *op_gstate.global_buffer;

	scan_chunk_idx = 0;
	scan_chunk_count = buf.ChunkCount();
	scan_chunk_done = 0;

	scan_chunks_per_thread = MaxValue<idx_t>((scan_chunk_count + op_gstate.num_threads - 1) / op_gstate.num_threads, 1);

	global_stage = ReservoirSourceStage::SCAN_BUFFER;
}

bool ReservoirGlobalSourceState::AssignTask(ReservoirGlobalOperatorState &gstate, ReservoirLocalSourceState &lstate) {
	D_ASSERT(lstate.TaskFinished());

	auto guard = Lock();
	switch (global_stage.load()) {
	case ReservoirSourceStage::SCAN_BUFFER:
		if (scan_chunk_idx != scan_chunk_count) {
			lstate.local_stage = global_stage;
			lstate.scan_chunk_idx_from = scan_chunk_idx;
			scan_chunk_idx = MinValue<idx_t>(scan_chunk_count, scan_chunk_idx + scan_chunks_per_thread);
			lstate.scan_chunk_idx_to = scan_chunk_idx;
			return true;
		}
		break;
	case ReservoirSourceStage::DONE:
		break;
	default:
		throw InternalException("Unexpected ReservoirSourceStage in AssignTask!");
	}
	return false;
}

ReservoirLocalSourceState::ReservoirLocalSourceState(const PhysicalReservoir &op,
                                                     const ReservoirGlobalOperatorState &op_gstate,
                                                     Allocator &allocator)
    : local_stage(ReservoirSourceStage::SCAN_BUFFER) {
}

void ReservoirLocalSourceState::ExecuteTask(ReservoirGlobalOperatorState &op_gstate, ReservoirGlobalSourceState &gstate,
                                            DataChunk &chunk) {
	switch (local_stage) {
	case ReservoirSourceStage::SCAN_BUFFER:
		ScanBuf(op_gstate, gstate, chunk);
		break;
	default:
		throw InternalException("Unexpected ReservoirSourceStage in ExecuteTask!");
	}
}

bool ReservoirLocalSourceState::TaskFinished() const {
	switch (local_stage) {
	case ReservoirSourceStage::SCAN_BUFFER:
		return scan_state == nullptr;
	default:
		throw InternalException("Unexpected ReservoirSourceStage in TaskFinished!");
	}
}

void ReservoirLocalSourceState::ScanBuf(ReservoirGlobalOperatorState &op_gstate, ReservoirGlobalSourceState &gstate,
                                        DataChunk &chunk) {
	D_ASSERT(local_stage == ReservoirSourceStage::SCAN_BUFFER);

	if (!scan_state) {
		scan_state = make_uniq<ReservoirScanState>();
		scan_state->chunk_idx = scan_chunk_idx_from;
	}

	op_gstate.global_buffer->FetchChunk(scan_state->chunk_idx++, chunk);

	if (scan_state->chunk_idx == scan_chunk_idx_to) {
		scan_state = nullptr;
		auto guard = gstate.Lock();
		gstate.scan_chunk_done += scan_chunk_idx_to - scan_chunk_idx_from;
	}
}

SourceResultType PhysicalReservoir::GetData(ExecutionContext &context, DataChunk &chunk,
                                            OperatorSourceInput &input) const {
	auto &op_gstate = op_state->Cast<ReservoirGlobalOperatorState>();
	auto &gstate = input.global_state.Cast<ReservoirGlobalSourceState>();
	auto &lstate = input.local_state.Cast<ReservoirLocalSourceState>();
	op_gstate.scanned_data = true;
	is_impounding = false;

	if (gstate.global_stage == ReservoirSourceStage::INIT) {
		gstate.Initialize(op_gstate);
	}

	// Any call to GetData must produce tuples, otherwise the pipeline executor thinks that we're done
	// Therefore, we loop until we've produced tuples, or until the operator is actually done
	if (gstate.global_stage != ReservoirSourceStage::DONE) {
		if (!lstate.TaskFinished() || gstate.AssignTask(op_gstate, lstate)) {
			lstate.ExecuteTask(op_gstate, gstate, chunk);
		} else {
			auto guard = gstate.Lock();
			gstate.global_stage = ReservoirSourceStage::DONE;
		}
	}

	return chunk.size() == 0 ? SourceResultType::FINISHED : SourceResultType::HAVE_MORE_OUTPUT;
}

double PhysicalReservoir::GetProgress(ClientContext &context, GlobalSourceState &gstate) const {
	return PhysicalOperator::GetProgress(context, gstate);
}

//===--------------------------------------------------------------------===//
// Pipeline Construction
//===--------------------------------------------------------------------===//
void PhysicalReservoir::BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) {
	op_state.reset();

	// todo: the pipeline dependency is wrong currently.

	// 'current' is the main pipeline: add this operator
	auto &state = meta_pipeline.GetState();
	state.AddPipelineOperator(current, *this);
	children[0]->BuildPipelines(current, meta_pipeline);

	// save the last added pipeline to set up dependencies later (in case we need to add a child pipeline)
	vector<shared_ptr<Pipeline>> pipelines_so_far;
	meta_pipeline.GetPipelines(pipelines_so_far, false);
	auto &last_pipeline = *pipelines_so_far.back();

	meta_pipeline.CreateChildPipeline(current, *this, last_pipeline);

	//	// 1. First Path: Source --> ... --> sink
	//	D_ASSERT(children.size() == 1);
	//
	//	// copy the pipeline
	//	auto &new_current = meta_pipeline.CreateUnionPipeline(current, false);
	//	// build the caching operator pipeline
	//	state.AddPipelineOperator(new_current, *this);
	//	children[0]->BuildPipelines(new_current, meta_pipeline);
	//
	//	// 2. Second Path: Source --> ... --> Reservoir, and Reservoir --> sink
	//	op_gstate.reset();
	//	D_ASSERT(children.size() == 1);
	//
	//	// single operator: the operator becomes the data source of the current pipeline
	//	state.SetPipelineSource(current, *this);
	//
	//	// we create a new pipeline starting from the child
	//	auto &child_meta_pipeline = meta_pipeline.CreateChildMetaPipeline(current, *this);
	//	child_meta_pipeline.Build(*children[0]);
}
} // namespace duckdb
