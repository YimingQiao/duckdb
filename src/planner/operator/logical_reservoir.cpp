#include "duckdb/planner/operator/logical_reservoir.hpp"

namespace duckdb {

idx_t LogicalReservoir::EstimateCardinality(ClientContext &context) {
	return LogicalOperator::EstimateCardinality(context);
}

} // namespace duckdb