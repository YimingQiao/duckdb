//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/optimizer/vector_filter_pull_up.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/main/client_context.hpp"
#include "duckdb/optimizer/rule.hpp"
#include "duckdb/parser/expression_map.hpp"
#include "duckdb/planner/logical_operator_visitor.hpp"

namespace duckdb {
class LogicalOperator;
class Optimizer;

class VectorFilterPullUp {
public:
	VectorFilterPullUp() {};

	unique_ptr<LogicalOperator> Rewrite(unique_ptr<LogicalOperator> op);

private:
};
} // namespace duckdb
