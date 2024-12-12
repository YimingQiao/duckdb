#include "duckdb/execution/operator/reservoir/physical_reservoir.hpp"

#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/execution/operator/reservoir/function_profiler.hpp"
#include "duckdb/main/query_profiler.hpp"
#include "duckdb/parallel/task_scheduler.hpp"
#include "duckdb/parallel/thread_context.hpp"
#include "duckdb/storage/buffer_manager.hpp"
#include "duckdb/storage/temporary_memory_manager.hpp"

namespace duckdb {
PhysicalReservoir::PhysicalReservoir(LogicalOperator &op, vector<LogicalType> types, idx_t estimated_cardinality)
    : PhysicalOperator(PhysicalOperatorType::RESERVOIR, std::move(types), estimated_cardinality), is_impounding(true) {
}

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
class ReservoirGlobalSinkState : public GlobalSinkState {
public:
	ReservoirGlobalSinkState(const PhysicalReservoir &op_p, ClientContext &context_p)
	    : context(context_p), op(op_p),
	      num_threads(NumericCast<idx_t>(TaskScheduler::GetScheduler(context).NumberOfThreads())),
	      buffer_manager(BufferManager::GetBufferManager(context)),
	      temporary_memory_state(TemporaryMemoryManager::Get(context).Register(context)), finalized(false) {
		global_buffer = make_uniq<ColumnDataCollection>(buffer_manager, op_p.types);
	}

public:
	ClientContext &context;
	const PhysicalReservoir &op;

	const idx_t num_threads;

	BufferManager &buffer_manager;
	//! Temporary memory state for managing this operator's memory usage
	unique_ptr<TemporaryMemoryState> temporary_memory_state;

	//! Global Buffer
	unique_ptr<ColumnDataCollection> global_buffer;

	//! Whether all data has been sinked
	bool finalized;
	//! The number of active local states
	atomic<idx_t> active_local_states;

	//! memory management
	idx_t total_size;

	//! Buffer held by each thread
	vector<unique_ptr<ColumnDataCollection>> local_buffers;
};

class ReservoirLocalSinkState : public LocalSinkState {
public:
	const idx_t CHUNK_THRESHOLD = DEFAULT_ROW_GROUP_SIZE / DEFAULT_STANDARD_VECTOR_SIZE;

public:
	ReservoirLocalSinkState(const PhysicalReservoir &op, ClientContext &context, ReservoirGlobalSinkState &gstate)
	    : num_chunk(0) {
		auto &buffer_manager = BufferManager::GetBufferManager(context);
		buffer = make_uniq<ColumnDataCollection>(buffer_manager, op.types);
		++gstate.active_local_states;
	}

public:
	//! Thread-local buffer
	unique_ptr<ColumnDataCollection> buffer;

	//! chunk number
	idx_t num_chunk;
};

unique_ptr<GlobalSinkState> PhysicalReservoir::GetGlobalSinkState(ClientContext &context) const {
	return make_uniq<ReservoirGlobalSinkState>(*this, context);
}

unique_ptr<LocalSinkState> PhysicalReservoir::GetLocalSinkState(ExecutionContext &context) const {
	auto &gstate = sink_state->Cast<ReservoirGlobalSinkState>();
	return make_uniq<ReservoirLocalSinkState>(*this, context.client, gstate);
}

SinkResultType PhysicalReservoir::Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const {
	auto start_time = std::chrono::high_resolution_clock::now();

	auto &gstate = input.global_state.Cast<ReservoirGlobalSinkState>();
	auto &lstate = input.local_state.Cast<ReservoirLocalSinkState>();

	lstate.buffer->Append(chunk);

	auto end_time = std::chrono::high_resolution_clock::now();
	uint64_t duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
	BeeProfiler::Get().InsertStatRecord("[PhysicalReservoir] Sink\t(0x" + std::to_string(uint64_t(this)) + ")",
	                                    duration_ns);

	return SinkResultType::NEED_MORE_INPUT;
}

SinkCombineResultType PhysicalReservoir::Combine(ExecutionContext &context, OperatorSinkCombineInput &input) const {
	auto &gstate = input.global_state.Cast<ReservoirGlobalSinkState>();
	auto &lstate = input.local_state.Cast<ReservoirLocalSinkState>();

	auto guard = gstate.Lock();
	gstate.local_buffers.push_back(std::move(lstate.buffer));
	if (gstate.local_buffers.size() == gstate.active_local_states) {
		// Set to 0 until PrepareFinalize
		gstate.temporary_memory_state->SetZero();
	}

	auto &client_profiler = QueryProfiler::Get(context.client);
	context.thread.profiler.Flush(*this);
	client_profiler.Flush(context.thread.profiler);
	return SinkCombineResultType::FINISHED;
}

//===--------------------------------------------------------------------===//
// Finalize
//===--------------------------------------------------------------------===//
static idx_t GetTupleWidth(const vector<LogicalType> &types, bool &all_constant) {
	idx_t tuple_width = 0;
	all_constant = true;
	for (auto &type : types) {
		tuple_width += GetTypeIdSize(type.InternalType());
		all_constant &= TypeIsConstantSize(type.InternalType());
	}
	return tuple_width + AlignValue(types.size()) / 8 + GetTypeIdSize(PhysicalType::UINT64);
}

void PhysicalReservoir::PrepareFinalize(ClientContext &context, GlobalSinkState &global_state) const {
	auto &gstate = global_state.Cast<ReservoirGlobalSinkState>();
	auto &global_buffer = *gstate.global_buffer;
	gstate.total_size = global_buffer.SizeInBytes();
	bool all_constant;
	gstate.temporary_memory_state->SetMaterializationPenalty(GetTupleWidth(children[0]->types, all_constant));
	gstate.temporary_memory_state->SetRemainingSize(gstate.total_size);
}

SinkFinalizeType PhysicalReservoir::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                             OperatorSinkFinalizeInput &input) const {
	auto &sink = input.global_state.Cast<ReservoirGlobalSinkState>();
	auto &global_buffer = *sink.global_buffer;

	// combine local buffers
	for (auto &local_buffer : sink.local_buffers) {
		global_buffer.Combine(*local_buffer);
	}

	sink.local_buffers.clear();
	sink.finalized = true;

	auto now = std::chrono::system_clock::now();
	auto duration = now.time_since_epoch();
	auto tick = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() % 1000000;
	std::cerr << "[PhysicalReservoir] (0x" << uint64_t(&sink.op) << ")\tSink Finalize Ends\tTicks: " << tick << "ms\n";

	return SinkFinalizeType::READY;
}

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
enum class ReservoirSourceStage : uint8_t { INIT, SCAN_BUFFER, DONE };

class ReservoirLocalSourceState;

class ReservoirGlobalSourceState : public GlobalSourceState {
public:
	ReservoirGlobalSourceState(const PhysicalReservoir &op, const ClientContext &context);

	//! Initialize this source state using the info in the sink
	void Initialize(ReservoirGlobalSinkState &sink);

	//! Prepare the scan_buf stage
	void PrepareScan(ReservoirGlobalSinkState &sink);

	//! Assigns a task to a local source state
	bool AssignTask(ReservoirGlobalSinkState &sink, ReservoirLocalSourceState &lstate);

	idx_t MaxThreads() override {
		D_ASSERT(op.sink_state);
		auto &gstate = op.sink_state->Cast<ReservoirGlobalSinkState>();

		idx_t count = gstate.global_buffer->Count() - scanned_row;
		std::cerr << "[PhysicalReservoir] (0x" + std::to_string(uint64_t(&op)) +
		                 ")\tSource Row Number: " + std::to_string(count) + "\n";

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
	idx_t scan_chunks_per_task = DConstants::INVALID_INDEX;

	idx_t parallel_scan_chunk_count;

	atomic<idx_t> scanned_row;
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
	ReservoirLocalSourceState(const PhysicalReservoir &op, const ReservoirGlobalSinkState &sink, Allocator &allocator);

	//! Do the work this thread has been assigned
	void ExecuteTask(ReservoirGlobalSinkState &sink, ReservoirGlobalSourceState &gstate, DataChunk &chunk);

	//! Whether this thread has finished the work it has been assigned
	bool TaskFinished() const;

	//! Scan
	void ScanBuf(ReservoirGlobalSinkState &sink, ReservoirGlobalSourceState &gstate, DataChunk &chunk);

public:
	//! The stage that this thread was assigned work for
	ReservoirSourceStage local_stage;

	idx_t scan_chunk_idx_from = DConstants::INVALID_INDEX;
	idx_t scan_chunk_idx_to = DConstants::INVALID_INDEX;

	unique_ptr<ReservoirScanState> scan_state;
};

shared_ptr<GlobalSourceState> PhysicalReservoir::GetGlobalSourceState(ClientContext &context) const {
	if (global_source_state == nullptr) {
		global_source_state = make_uniq<ReservoirGlobalSourceState>(*this, context);
	}
	return global_source_state;
}

unique_ptr<LocalSourceState> PhysicalReservoir::GetLocalSourceState(ExecutionContext &context,
                                                                    GlobalSourceState &gstate) const {
	return make_uniq<ReservoirLocalSourceState>(*this, sink_state->Cast<ReservoirGlobalSinkState>(),
	                                            BufferAllocator::Get(context.client));
}

ReservoirGlobalSourceState::ReservoirGlobalSourceState(const PhysicalReservoir &op, const ClientContext &context)
    : op(op), global_stage(ReservoirSourceStage::INIT), parallel_scan_chunk_count(1), scanned_row(0) {
}

void ReservoirGlobalSourceState::Initialize(ReservoirGlobalSinkState &sink) {
	auto guard = Lock();
	if (global_stage != ReservoirSourceStage::INIT) {
		// Another thread initialized
		return;
	}

	PrepareScan(sink);
}

void ReservoirGlobalSourceState::PrepareScan(ReservoirGlobalSinkState &sink) {
	D_ASSERT(global_stage != ReservoirSourceStage::SCAN_BUFFER);
	auto &buf = *sink.global_buffer;

	scan_chunk_idx = 0;
	scan_chunk_count = buf.ChunkCount();
	scan_chunk_done = 0;
	scanned_row = 0;

	scan_chunks_per_task = DEFAULT_ROW_GROUP_SIZE / DEFAULT_STANDARD_VECTOR_SIZE;

	global_stage = ReservoirSourceStage::SCAN_BUFFER;
}

bool ReservoirGlobalSourceState::AssignTask(ReservoirGlobalSinkState &sink, ReservoirLocalSourceState &lstate) {
	D_ASSERT(lstate.TaskFinished());

	auto guard = Lock();
	switch (global_stage.load()) {
	case ReservoirSourceStage::SCAN_BUFFER:
		if (scan_chunk_idx != scan_chunk_count) {
			lstate.local_stage = global_stage;
			lstate.scan_chunk_idx_from = scan_chunk_idx;
			scan_chunk_idx = MinValue<idx_t>(scan_chunk_count, scan_chunk_idx + scan_chunks_per_task);
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

ReservoirLocalSourceState::ReservoirLocalSourceState(const PhysicalReservoir &op, const ReservoirGlobalSinkState &sink,
                                                     Allocator &allocator)
    : local_stage(ReservoirSourceStage::SCAN_BUFFER) {
}

void ReservoirLocalSourceState::ExecuteTask(ReservoirGlobalSinkState &sink, ReservoirGlobalSourceState &gstate,
                                            DataChunk &chunk) {
	switch (local_stage) {
	case ReservoirSourceStage::SCAN_BUFFER:
		ScanBuf(sink, gstate, chunk);
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

void ReservoirLocalSourceState::ScanBuf(ReservoirGlobalSinkState &sink, ReservoirGlobalSourceState &gstate,
                                        DataChunk &chunk) {
	D_ASSERT(local_stage == ReservoirSourceStage::SCAN_BUFFER);

	if (!scan_state) {
		scan_state = make_uniq<ReservoirScanState>();
		scan_state->chunk_idx = scan_chunk_idx_from;
	}

	sink.global_buffer->FetchChunk(scan_state->chunk_idx++, chunk);

	if (scan_state->chunk_idx == scan_chunk_idx_to) {
		scan_state = nullptr;
		auto guard = gstate.Lock();
		gstate.scan_chunk_done += scan_chunk_idx_to - scan_chunk_idx_from;
	}
}

SourceResultType PhysicalReservoir::GetData(ExecutionContext &context, DataChunk &chunk,
                                            OperatorSourceInput &input) const {
	auto start_time = std::chrono::high_resolution_clock::now();

	auto &sink = sink_state->Cast<ReservoirGlobalSinkState>();
	auto &gstate = input.global_state.Cast<ReservoirGlobalSourceState>();
	auto &lstate = input.local_state.Cast<ReservoirLocalSourceState>();

	if (gstate.global_stage == ReservoirSourceStage::INIT) {
		gstate.Initialize(sink);
	}

	if (!lstate.TaskFinished()) {
		lstate.ExecuteTask(sink, gstate, chunk);
	} else {
		if (gstate.AssignTask(sink, lstate)) {
			lstate.ExecuteTask(sink, gstate, chunk);
		} else if (gstate.scan_chunk_count == gstate.scan_chunk_done) {
			gstate.global_stage = ReservoirSourceStage::DONE;
		}
	}

	gstate.scanned_row += chunk.size();

	auto end_time = std::chrono::high_resolution_clock::now();
	uint64_t duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
	BeeProfiler::Get().InsertStatRecord("[PhysicalReservoir] GetData\t(0x" + std::to_string(uint64_t(this)) + ")",
	                                    duration_ns);

	return chunk.size() == 0 ? SourceResultType::FINISHED : SourceResultType::HAVE_MORE_OUTPUT;
}

double PhysicalReservoir::GetProgress(ClientContext &context, GlobalSourceState &gstate) const {
	return PhysicalOperator::GetProgress(context, gstate);
}

//===--------------------------------------------------------------------===//
// Pipeline Construction
//===--------------------------------------------------------------------===//
void PhysicalReservoir::BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) {
	D_ASSERT(children.size() == 1);

	if (BLOCKED) {
		// it is a sink
		sink_state.reset();

		auto &state = meta_pipeline.GetState();

		// Second Pipeline: Reservoir --> ... --> Sink
		state.SetPipelineSource(current, *this);

		// Third Pipeline: Source --> ... --> Reservoir
		auto &child_meta_pipeline = meta_pipeline.CreateChildMetaPipeline(current, *this);
		child_meta_pipeline.Build(*children[0]);

		return;
	}

	op_state.reset();
	sink_state.reset();

	// ------------------------------ First Pipeline Group: ... --> ... --> Reservoir ------------------------------
	reservoir_meta_pipeline = &meta_pipeline.CreateChildMetaPipeline(current, *this);
	reservoir_meta_pipeline->Build(*children[0]);

	// ------------------------------ Second Pipeline Group: Reservoir --> ... --> ... ------------------------------
	auto &state = meta_pipeline.GetState();
	state.SetPipelineSource(current, *this);

	// ------------------------------ Third Pipeline Group: ... --> Reservoir --> ... ------------------------------
	auto &operator_pipeline = meta_pipeline.CreateUnionPipeline(current, false);

	PhysicalOperator *op = children[0].get();
	while (op->type != PhysicalOperatorType::RESERVOIR) {
		if (op->children.empty()) {
			state.SetPipelineSource(operator_pipeline, *op);
			return;
		}

		state.AddPipelineOperator(operator_pipeline, *op);
		op = op->children[0].get();
	}
	static_cast<PhysicalReservoir *>(op)->BuildReservoirPath(operator_pipeline, meta_pipeline);
}

void PhysicalReservoir::BuildReservoirPath(duckdb::Pipeline &current, duckdb::MetaPipeline &meta_pipeline) {
	D_ASSERT(children.size() == 1);

	current.AddDependency(reservoir_meta_pipeline->GetBasePipeline());

	// ------------------------------ Second Pipeline Group: Reservoir --> ... --> ... ------------------------------
	auto &state = meta_pipeline.GetState();
	state.SetPipelineSource(current, *this);

	// ------------------------------ Third Pipeline Group: ... --> Reservoir --> ... ------------------------------
	auto &operator_pipeline = meta_pipeline.CreateUnionPipeline(current, true);

	PhysicalOperator *op = children[0].get();
	while (op->type != PhysicalOperatorType::RESERVOIR) {
		if (op->children.empty()) {
			state.SetPipelineSource(operator_pipeline, *op);
			return;
		}

		state.AddPipelineOperator(operator_pipeline, *op);
		op = op->children[0].get();
	}
	static_cast<PhysicalReservoir *>(op)->BuildReservoirPath(operator_pipeline, meta_pipeline);
}
} // namespace duckdb
