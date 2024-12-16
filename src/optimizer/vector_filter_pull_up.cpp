#include "duckdb/optimizer/vector_filter_pull_up.hpp"

namespace duckdb {
unique_ptr<LogicalOperator> VectorFilterPullUp::Rewrite(unique_ptr<LogicalOperator> op) {
	return op;
}
} // namespace duckdb
