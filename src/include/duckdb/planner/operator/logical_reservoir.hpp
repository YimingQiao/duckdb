//===----------------------------------------------------------------------===//
//                         ReservoirDuckDB
//
// duckdb/planner/operator/logical_reservoir.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/planner/bound_query_node.hpp"
#include "duckdb/planner/logical_operator.hpp"

namespace duckdb {

//! LogicalTopN represents a comibination of ORDER BY and LIMIT clause, using Min/Max Heap
class LogicalReservoir : public LogicalOperator {
public:
	static constexpr const LogicalOperatorType TYPE = LogicalOperatorType::LOGICAL_RESERVOIR;

public:
	LogicalReservoir()
	    : LogicalOperator(LogicalOperatorType::LOGICAL_RESERVOIR) {
	}

public:
	vector<ColumnBinding> GetColumnBindings() override {
		return children[0]->GetColumnBindings();
	}

	idx_t EstimateCardinality(ClientContext &context) override;

protected:
	void ResolveTypes() override {
		types = children[0]->types;
	}
};
} // namespace duckdb
