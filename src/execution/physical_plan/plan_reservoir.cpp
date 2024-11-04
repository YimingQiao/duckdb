#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/planner/operator/logical_reservoir.hpp"
#include "duckdb/execution/operator/reservoir/physical_reservoir.hpp"

namespace duckdb {

unique_ptr<PhysicalOperator> PhysicalPlanGenerator::CreatePlan(LogicalReservoir &op) {
	D_ASSERT(op.children.size() == 1);
	auto plan = CreatePlan(*op.children[0]);
	auto reservoir = make_uniq<PhysicalReservoir>(op, op.types, op.estimated_cardinality);
    reservoir->children.push_back(std::move(plan));
    return std::move(reservoir);
}
}