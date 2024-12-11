//===----------------------------------------------------------------------===//
//                         ReservoirDuckDB
//
// duckdb/execution/operator/reservoir/physical_reservoir.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/planner/expression.hpp"
#include "duckdb/common/enums/operator_result_type.hpp"
#include "duckdb/parallel/meta_pipeline.hpp"

namespace duckdb {

// The Reservoir Operator is used to store data before probing.
// 1. When hash table building is not finished, the reservoir works as a sink.
// 2. When hash table building finishes, the left pipeline begins, calling the executeInternal function of this
// reservoir.
//    Thus, the reservoir stops sink, and then execute as a normal operator.
// 3. When the all data from source is probed. The reservoir works as a source, outputting its stored data.
//
// I think it is a very clean design.
class PhysicalReservoir : public PhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::RESERVOIR;

	static constexpr const bool BLOCKED = false;

public:
	PhysicalReservoir(LogicalOperator &op, vector<LogicalType> types, idx_t estimated_cardinality);

	mutable atomic<bool> is_impounding;

public:
	// Operator interface
	OperatorResultType Execute(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
	                           GlobalOperatorState &gstate, OperatorState &state) const override;

	bool ParallelOperator() const override {
		return true;
	}

public:
	// Source interface
	unique_ptr<GlobalSourceState> GetGlobalSourceState(ClientContext &context) const override;
	unique_ptr<LocalSourceState> GetLocalSourceState(ExecutionContext &context,
	                                                 GlobalSourceState &gstate) const override;
	SourceResultType GetData(ExecutionContext &context, DataChunk &chunk, OperatorSourceInput &input) const override;

	double GetProgress(ClientContext &context, GlobalSourceState &gstate) const override;

	bool IsSource() const override {
		return true;
	}
	bool ParallelSource() const override {
		return true;
	}

public:
	// Sink Interface
	unique_ptr<GlobalSinkState> GetGlobalSinkState(ClientContext &context) const override;

	unique_ptr<LocalSinkState> GetLocalSinkState(ExecutionContext &context) const override;
	SinkResultType Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const override;
	SinkCombineResultType Combine(ExecutionContext &context, OperatorSinkCombineInput &input) const override;
	void PrepareFinalize(ClientContext &context, GlobalSinkState &global_state) const override;
	SinkFinalizeType Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
	                          OperatorSinkFinalizeInput &input) const override;

	bool IsSink() const override {
		return true;
	}
	bool ParallelSink() const override {
		return true;
	}

public:
	void BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) override;

	MetaPipeline *reservoir_meta_pipeline;

private:
	void BuildReservoirPath(Pipeline &current, MetaPipeline &meta_pipeline);
};
} // namespace duckdb
