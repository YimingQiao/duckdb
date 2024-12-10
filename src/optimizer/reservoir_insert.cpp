#include "duckdb/optimizer/reservoir_insert.hpp"

#include "duckdb/main/client_config.hpp"
#include "duckdb/planner/operator/logical_reservoir.hpp"

namespace duckdb {
unique_ptr<LogicalOperator> ReservoirInsert::Rewrite(unique_ptr<LogicalOperator> op) {
	for (idx_t i = 0; i < op->children.size(); i++) {
		op->children[i] = Rewrite(std::move(op->children[i]));
	}

	switch (op->type) {
	case LogicalOperatorType::LOGICAL_COMPARISON_JOIN: {
		return DoInsert(unique_ptr_cast<LogicalOperator, LogicalComparisonJoin>(std::move(op)));
	}
	default:
		return op;
	}
}

bool HasEquality(vector<JoinCondition> &conds, idx_t &range_count) {
	for (size_t c = 0; c < conds.size(); ++c) {
		auto &cond = conds[c];
		switch (cond.comparison) {
		case ExpressionType::COMPARE_EQUAL:
		case ExpressionType::COMPARE_NOT_DISTINCT_FROM:
			return true;
		case ExpressionType::COMPARE_LESSTHAN:
		case ExpressionType::COMPARE_GREATERTHAN:
		case ExpressionType::COMPARE_LESSTHANOREQUALTO:
		case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
			++range_count;
			break;
		case ExpressionType::COMPARE_NOTEQUAL:
		case ExpressionType::COMPARE_DISTINCT_FROM:
			break;
		default:
			throw NotImplementedException("Unimplemented comparison join");
		}
	}
	return false;
}

unique_ptr<LogicalOperator> ReservoirInsert::DoInsert(unique_ptr<LogicalComparisonJoin> op) {
	if (op->children[0]->type == LogicalOperatorType::LOGICAL_GET ||
	    op->children[0]->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
		return std::move(op);
	}

	idx_t has_range = 0;
	bool has_equality = HasEquality(op->conditions, has_range);

	bool can_iejoin = has_range >= 2;
	switch (op->join_type) {
	case JoinType::SEMI:
	case JoinType::ANTI:
	case JoinType::RIGHT_ANTI:
	case JoinType::RIGHT_SEMI:
	case JoinType::MARK:
		can_iejoin = false;
		break;
	default:
		break;
	}

	auto &client_config = ClientConfig::GetConfig(context);

	//	TODO: Extend PWMJ to handle all comparisons and projection maps
	const auto prefer_range_joins = client_config.prefer_range_joins && can_iejoin;

	if (has_equality && !prefer_range_joins) {
		auto reservoir = make_uniq<LogicalReservoir>();
		reservoir->children.push_back(std::move(op->children[0]));
		op->children[0] = std::move(reservoir);
	}

	return std::move(op);
}
} // namespace duckdb
