#include "duckdb/execution/operator/reservoir/physical_reservoir.hpp"
#include "duckdb/parallel/task_scheduler.hpp"
#include "duckdb/storage/temporary_memory_manager.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/main/query_profiler.hpp"
#include "duckdb/parallel/thread_context.hpp"
#include "duckdb/storage/buffer_manager.hpp"

namespace duckdb {

PhysicalReservoir::PhysicalReservoir(LogicalOperator &op, vector<LogicalType> types, idx_t estimated_cardinality)
    : CachingPhysicalOperator(type, op.types, estimated_cardinality) {
    impoundment = new bool(true);
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
		buffer = make_uniq<ColumnDataCollection>(buffer_manager, op_p.types);
	}

	void ScheduleFinalize(Pipeline &pipeline, Event &event);
	void InitializeProbeSpill();

public:
	ClientContext &context;
	const PhysicalReservoir &op;

	const idx_t num_threads;

    BufferManager &buffer_manager;
	//! Temporary memory state for managing this operator's memory usage
	unique_ptr<TemporaryMemoryState> temporary_memory_state;

	//! Global HT used by the join
	unique_ptr<ColumnDataCollection> buffer;

	//! Whether or not the hash table has been finalized
	bool finalized;
	//! The number of active local states
	atomic<idx_t> active_local_states;

	//! Whether we are doing an external + some sizes
	// bool external;
	// idx_t total_size;
	// idx_t max_partition_size;
	// idx_t max_partition_count;

	//! Hash tables built by each thread
	vector<unique_ptr<ColumnDataCollection>> local_buffers;

    //! Whether or not we have started scanning data using GetData
	atomic<bool> scanned_data;
};

class ReservoirLocalSinkState : public LocalSinkState {
public:
	ReservoirLocalSinkState(const PhysicalReservoir &op, ClientContext &context, ReservoirGlobalSinkState &gstate) {
		auto &buffer_manager = BufferManager::GetBufferManager(context);
		buffer = make_uniq<ColumnDataCollection>(buffer_manager, op.types);
		gstate.active_local_states++;
	}

public:
	//! Thread-local buffer
	unique_ptr<ColumnDataCollection> buffer;
};

unique_ptr<GlobalSinkState> PhysicalReservoir::GetGlobalSinkState(ClientContext &context) const {
	return make_uniq<ReservoirGlobalSinkState>(*this, context);
}

unique_ptr<LocalSinkState> PhysicalReservoir::GetLocalSinkState(ExecutionContext &context) const {
	auto &gstate = sink_state->Cast<ReservoirGlobalSinkState>();
	return make_uniq<ReservoirLocalSinkState>(*this, context.client, gstate);
}

SinkResultType PhysicalReservoir::Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const {
    if (*impoundment) {
        auto &lstate = input.local_state.Cast<ReservoirLocalSinkState>();
        lstate.buffer->Append(chunk);
        return SinkResultType::NEED_MORE_INPUT;
    }
    return SinkResultType::FINISHED;
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
	// auto &gstate = global_state.Cast<ReservoirGlobalSinkState>();
	// auto &ht = *gstate.buffer;
	// gstate.total_size = ht.GetTotalSize(gstate.local_hash_tables, gstate.max_partition_size, gstate.max_partition_count);
	// bool all_constant;
	// gstate.temporary_memory_state->SetMaterializationPenalty(GetTupleWidth(children[0]->types, all_constant));
	// gstate.temporary_memory_state->SetRemainingSize(gstate.total_size);
}

SinkFinalizeType PhysicalReservoir::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                             OperatorSinkFinalizeInput &input) const {
	auto &sink = input.global_state.Cast<ReservoirGlobalSinkState>();
	auto &buf = *sink.buffer;

	// sink.temporary_memory_state->UpdateReservation(context);
	// sink.external = sink.temporary_memory_state->GetReservation() < sink.total_size;
	// if (sink.external) {
	// 	// External Hash Join
	// 	sink.perfect_join_executor.reset();

	// 	const auto max_partition_ht_size =
	// 	    sink.max_partition_size + JoinHashTable::PointerTableSize(sink.max_partition_count);
	// 	if (max_partition_ht_size > sink.temporary_memory_state->GetReservation()) {
	// 		// We have to repartition
	// 		ht.SetRepartitionRadixBits(sink.temporary_memory_state->GetReservation(), sink.max_partition_size,
	// 		                           sink.max_partition_count);
	// 		auto new_event = make_shared_ptr<HashJoinRepartitionEvent>(pipeline, *this, sink, sink.local_hash_tables);
	// 		event.InsertEvent(std::move(new_event));
	// 	} else {
	// 		// No repartitioning! We do need some space for partitioning the probe-side, though
	// 		const auto probe_side_requirement =
	// 		    GetPartitioningSpaceRequirement(context, children[0]->types, ht.GetRadixBits(), sink.num_threads);
	// 		sink.temporary_memory_state->SetMinimumReservation(max_partition_ht_size + probe_side_requirement);
	// 		for (auto &local_ht : sink.local_hash_tables) {
	// 			ht.Merge(*local_ht);
	// 		}
	// 		sink.local_hash_tables.clear();
	// 		sink.hash_table->PrepareExternalFinalize(sink.temporary_memory_state->GetReservation());
	// 		sink.ScheduleFinalize(pipeline, event);
	// 	}
	// 	sink.finalized = true;
	// 	return SinkFinalizeType::READY;
	// }

	// In-memory Hash Join
	for (auto &local_buf : sink.local_buffers) {
		buf.Combine(*local_buf);
	}

	sink.local_buffers.clear();

	sink.finalized = true;
	return SinkFinalizeType::READY;
}
//===--------------------------------------------------------------------===//
// Operator
//===--------------------------------------------------------------------===//
unique_ptr<OperatorState> PhysicalReservoir::GetOperatorState(ExecutionContext &context) const {
    auto state = make_uniq<OperatorState>();
    return state;
}

OperatorResultType PhysicalReservoir::ExecuteInternal(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
	                                                  GlobalOperatorState &gstate, OperatorState &state) const {
    *impoundment = false;
    chunk.Reference(input);
    return OperatorResultType::NEED_MORE_INPUT;
}

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
enum class ReservoirSourceStage : uint8_t { INIT, SCAN_BUF, DONE };

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

		idx_t count;
		idx_t count = gstate.buffer->Count();
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
	ReservoirScanState()
	    : chunk_idx(DConstants::INVALID_INDEX) {
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

unique_ptr<GlobalSourceState> PhysicalReservoir::GetGlobalSourceState(ClientContext &context) const {
    return make_uniq<ReservoirGlobalSourceState>(*this, context);
}

unique_ptr<LocalSourceState> PhysicalReservoir::GetLocalSourceState(ExecutionContext &context,
                                                                   GlobalSourceState &gstate) const {
	return make_uniq<ReservoirLocalSourceState>(*this, sink_state->Cast<ReservoirGlobalSinkState>(),
	                                            BufferAllocator::Get(context.client));
}

ReservoirGlobalSourceState::ReservoirGlobalSourceState(const PhysicalReservoir &op, const ClientContext &context)
    : op(op), global_stage(ReservoirSourceStage::INIT), parallel_scan_chunk_count(1) {
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
    D_ASSERT(global_stage != ReservoirSourceStage::SCAN_BUF);
    auto &buf = *sink.buffer;

    scan_chunk_idx = 0;
	scan_chunk_count = buf.ChunkCount();
	scan_chunk_done = 0;

	scan_chunks_per_thread = MaxValue<idx_t>((scan_chunk_count + sink.num_threads - 1) / sink.num_threads, 1);

    global_stage = ReservoirSourceStage::SCAN_BUF;
}

bool ReservoirGlobalSourceState::AssignTask(ReservoirGlobalSinkState &sink, ReservoirLocalSourceState &lstate) {
	D_ASSERT(lstate.TaskFinished());

	auto guard = Lock();
	switch (global_stage.load()) {
	case ReservoirSourceStage::SCAN_BUF:
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
		throw InternalException("Unexpected HashJoinSourceStage in AssignTask!");
	}
	return false;
}

ReservoirLocalSourceState::ReservoirLocalSourceState(const PhysicalReservoir &op, const ReservoirGlobalSinkState &sink,
                                                     Allocator &allocator)
    : local_stage(ReservoirSourceStage::SCAN_BUF) { }

void ReservoirLocalSourceState::ExecuteTask(ReservoirGlobalSinkState &sink, ReservoirGlobalSourceState &gstate,
                                           DataChunk &chunk) {
	switch (local_stage) {
	case ReservoirSourceStage::SCAN_BUF:
		ScanBuf(sink, gstate, chunk);
		break;
	default:
		throw InternalException("Unexpected HashJoinSourceStage in ExecuteTask!");
	}
}

bool ReservoirLocalSourceState::TaskFinished() const {
	switch (local_stage) {
	case ReservoirSourceStage::SCAN_BUF:
		return scan_state == nullptr;
	default:
		throw InternalException("Unexpected HashJoinSourceStage in TaskFinished!");
	}
}

void ReservoirLocalSourceState::ScanBuf(ReservoirGlobalSinkState &sink, ReservoirGlobalSourceState &gstate,
                                        DataChunk &chunk) {
	D_ASSERT(local_stage == ReservoirSourceStage::SCAN_BUF);

	if (!scan_state) {
		scan_state = make_uniq<ReservoirScanState>();
        scan_state->chunk_idx = scan_chunk_idx_from;
	}

	sink.buffer->FetchChunk(scan_state->chunk_idx++, chunk);

	if (scan_state->chunk_idx == scan_chunk_idx_to) {
		scan_state = nullptr;
		auto guard = gstate.Lock();
		gstate.scan_chunk_done += scan_chunk_idx_to - scan_chunk_idx_from;
	}
}

SourceResultType PhysicalReservoir::GetData(ExecutionContext &context, DataChunk &chunk,
                                            OperatorSourceInput &input) const {
	auto &sink = sink_state->Cast<ReservoirGlobalSinkState>();
	auto &gstate = input.global_state.Cast<ReservoirGlobalSourceState>();
	auto &lstate = input.local_state.Cast<ReservoirLocalSourceState>();
	sink.scanned_data = true;

	if (gstate.global_stage == ReservoirSourceStage::INIT) {
		gstate.Initialize(sink);
	}

	// Any call to GetData must produce tuples, otherwise the pipeline executor thinks that we're done
	// Therefore, we loop until we've produced tuples, or until the operator is actually done
	if (gstate.global_stage != ReservoirSourceStage::DONE) {
		if (!lstate.TaskFinished() || gstate.AssignTask(sink, lstate)) {
			lstate.ExecuteTask(sink, gstate, chunk);
		} else {
            auto guard = gstate.Lock();
            gstate.global_stage = ReservoirSourceStage::DONE;
        }
	}

	return chunk.size() == 0 ? SourceResultType::FINISHED : SourceResultType::HAVE_MORE_OUTPUT;
}

//===--------------------------------------------------------------------===//
// Pipeline Construction
//===--------------------------------------------------------------------===//
void PhysicalReservoir::BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) {
	op_state.reset();

	auto &state = meta_pipeline.GetState();
	if (children.size() != 1) {
		throw InternalException("Operator not supported in BuildPipelines");
	}
	
	// copy the pipeline
	auto &new_current = meta_pipeline.CreateUnionPipeline(current, false);
	// build the caching operator pipeline
	state.AddPipelineOperator(new_current, *this);
	children[0]->BuildPipelines(new_current, meta_pipeline);

	// build the sink pipeline
	sink_state.reset();
	D_ASSERT(children.size() == 1);

	// single operator: the operator becomes the data source of the current pipeline
	state.SetPipelineSource(current, *this);

	// we create a new pipeline starting from the child
	auto &child_meta_pipeline = meta_pipeline.CreateChildMetaPipeline(current, *this);
	child_meta_pipeline.Build(*children[0]);
}
}