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
// I think it is a very clear design.
class PhysicalReservoir : public PhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::RESERVOIR;

public:
	PhysicalReservoir(LogicalOperator &op, vector<LogicalType> types, idx_t estimated_cardinality);

	// we can modify the variable is_impounding in a const function, i.e. Execute() const.
	mutable atomic<bool> is_impounding;

	const bool reservoir_debug = false;

public:
	// Operator interface
	unique_ptr<OperatorState> GetOperatorState(ExecutionContext &context) const override;
	unique_ptr<GlobalOperatorState> GetGlobalOperatorState(ClientContext &context) const override;
	OperatorResultType Execute(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
	                           GlobalOperatorState &gstate, OperatorState &state) const override;
	OperatorFinalizeResultType FinalExecute(ExecutionContext &context, DataChunk &chunk, GlobalOperatorState &gstate,
	                                        OperatorState &state_p) const override;

	bool ParallelOperator() const override {
		return true;
	}

	bool RequiresFinalExecute() const override {
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
	//! It builds two path
	//! 1. source --> ... --> sink
	//!    This pipeline has two stages, controlled by the variable is_impound.
	//! 2. Reservoir --> Sink
	void BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) override;
};
} // namespace duckdb
