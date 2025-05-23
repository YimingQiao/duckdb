#include "duckdb/optimizer/predicate_transfer/table_operator_namager.hpp"

#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"

namespace duckdb {
vector<reference<LogicalOperator>> TableOperatorManager::ExtractOperators(LogicalOperator &plan) {
	vector<reference<LogicalOperator>> ret;
	ExtractOperatorsInternal(plan, ret);
	SortTableOperators();
	return ret;
}

void TableOperatorManager::SortTableOperators() {
	sorted_table_operators.clear();
	for (auto &node : table_operators) {
		sorted_table_operators.emplace_back(node.second);
	}
	sort(sorted_table_operators.begin(), sorted_table_operators.end(),
	     [&](LogicalOperator *a, LogicalOperator *b) { return a->estimated_cardinality < b->estimated_cardinality; });
}

LogicalOperator *TableOperatorManager::GetTableOperator(idx_t table_idx) {
	auto itr = table_operators.find(table_idx);
	if (itr == table_operators.end()) {
		return nullptr;
	}

	return itr->second;
}

idx_t TableOperatorManager::GetTableOperatorOrder(const LogicalOperator *node) {
	if (sorted_table_operators.empty()) {
		SortTableOperators();
	}

	for (idx_t i = 0; i < sorted_table_operators.size(); i++) {
		if (sorted_table_operators[i] == node) {
			return i;
		}
	}
	return static_cast<idx_t>(-1); // fallback if not found
}

ColumnBinding TableOperatorManager::GetRenaming(ColumnBinding binding) {
	auto itr = rename_col_bindings.find(binding);
	while (itr != rename_col_bindings.end()) {
		binding = itr->second;
		itr = rename_col_bindings.find(binding);
	}
	return binding;
}

idx_t TableOperatorManager::GetScalarTableIndex(LogicalOperator *op) {
	switch (op->type) {
	case LogicalOperatorType::LOGICAL_CHUNK_GET:
	case LogicalOperatorType::LOGICAL_GET:
	case LogicalOperatorType::LOGICAL_DELIM_GET:
	case LogicalOperatorType::LOGICAL_PROJECTION:
	case LogicalOperatorType::LOGICAL_UNION:
	case LogicalOperatorType::LOGICAL_EXCEPT:
	case LogicalOperatorType::LOGICAL_INTERSECT: {
		return op->GetTableIndex()[0];
	}
	case LogicalOperatorType::LOGICAL_FILTER: {
		if (op->children[0]->type == LogicalOperatorType::LOGICAL_GET) {
			return op->children[0]->Cast<LogicalGet>().GetTableIndex()[0];
		}
		if (op->children[0]->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
			// For In-Clause optimization
			return op->children[0]->children[0]->Cast<LogicalGet>().GetTableIndex()[0];
		}
		if (op->children[0]->type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
			return op->children[0]->Cast<LogicalAggregate>().GetTableIndex()[0];
		}
		return op->Cast<LogicalGet>().GetTableIndex()[0];
	}
	case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY: {
		return op->GetTableIndex()[1];
	}
	default:
		return std::numeric_limits<idx_t>::max();
	}
}

bool TableOperatorManager::OperatorNeedsRelation(LogicalOperatorType op_type) {
	switch (op_type) {
	case LogicalOperatorType::LOGICAL_EXPRESSION_GET:
	case LogicalOperatorType::LOGICAL_GET:
	case LogicalOperatorType::LOGICAL_DELIM_GET:
	case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY:
	case LogicalOperatorType::LOGICAL_WINDOW:
		return true;
	default:
		return false;
	}
}

void TableOperatorManager::AddTableOperator(LogicalOperator *op) {
	op->estimated_cardinality = op->EstimateCardinality(context);

	idx_t table_idx = GetScalarTableIndex(op);
	if (table_idx != std::numeric_limits<idx_t>::max() && table_operators.find(table_idx) == table_operators.end()) {
		table_operators[table_idx] = op;
	}
}

void TableOperatorManager::ExtractOperatorsInternal(LogicalOperator &plan, vector<reference<LogicalOperator>> &joins) {
	LogicalOperator *op = &plan;
	while (op->children.size() == 1 && !OperatorNeedsRelation(op->type)) {
		if (op->type == LogicalOperatorType::LOGICAL_FILTER) {
			LogicalOperator *child = op->children[0].get();
			if (child->type == LogicalOperatorType::LOGICAL_GET) {
				AddTableOperator(op);
				return;
			}

			// TODO: support mark join
			// if (op->expressions.size() == 1 && op->expressions[0]->type == ExpressionType::BOUND_COLUMN_REF &&
			//     child->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
			// 	auto &join = child->Cast<LogicalComparisonJoin>();
			//
			// 	// if it is a mark join and the left table is small, we can build a BF based on it.
			// 	if (join.join_type == JoinType::MARK && join.children[1]->estimated_cardinality < 1000) {
			// 		AddTableOperator(op);
			// 	}
			// }

			D_ASSERT(!op->expressions.empty());
			ExtractOperatorsInternal(*child, joins);
			return;
		}

		if (op->type == LogicalOperatorType::LOGICAL_PROJECTION) {
			LogicalOperator *child = op->children[0].get();

			// TODO: support mark join
			// if (op->expressions.size() == 1 && op->expressions[0]->type == ExpressionType::BOUND_COLUMN_REF &&
			//     child->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
			// 	auto &join = child->Cast<LogicalComparisonJoin>();
			//
			// 	// if it is a mark join and the left table is small, we can build a BF based on it.
			// 	if (join.join_type == JoinType::MARK && join.children[1]->estimated_cardinality < 1000) {
			// 		AddTableOperator(op);
			// 	}
			// }

			D_ASSERT(!op->expressions.empty());
			ExtractOperatorsInternal(*child, joins);
			return;
		}
		op = op->children[0].get();
	}

	if (op->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN ||
	    op->type == LogicalOperatorType::LOGICAL_DELIM_JOIN) {
		auto &join = op->Cast<LogicalComparisonJoin>();
		switch (join.join_type) {
		case JoinType::MARK: {
			return;
		}
		case JoinType::INNER:
		case JoinType::LEFT:
		case JoinType::RIGHT:
		case JoinType::SEMI:
		case JoinType::RIGHT_SEMI: {
			if (std::any_of(join.conditions.begin(), join.conditions.end(), [](const JoinCondition &jc) {
				    return jc.comparison == ExpressionType::COMPARE_EQUAL &&
				           jc.left->type == ExpressionType::BOUND_COLUMN_REF &&
				           jc.right->type == ExpressionType::BOUND_COLUMN_REF;
			    })) {
				joins.push_back(*op);
			}
			break;
		}
		default:
			break;
		}
	}

	switch (op->type) {
	case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY: {
		auto &agg = op->Cast<LogicalAggregate>();
		if (agg.groups.empty() && agg.grouping_sets.size() <= 1) {
			ExtractOperatorsInternal(*op->children[0], joins);
			AddTableOperator(op);
		} else {
			for (size_t i = 0; i < agg.groups.size(); i++) {
				if (agg.groups[i]->type == ExpressionType::BOUND_COLUMN_REF) {
					auto &col_ref = agg.groups[i]->Cast<BoundColumnRefExpression>();
					rename_col_bindings.insert({agg.GetColumnBindings()[i], col_ref.binding});
				}
			}
			ExtractOperatorsInternal(*op->children[0], joins);
		}
		return;
	}
	case LogicalOperatorType::LOGICAL_PROJECTION: {
		for (size_t i = 0; i < op->expressions.size(); i++) {
			if (op->expressions[i]->type == ExpressionType::BOUND_COLUMN_REF) {
				auto &col_ref = op->expressions[i]->Cast<BoundColumnRefExpression>();
				rename_col_bindings.insert({op->GetColumnBindings()[i], col_ref.binding});
			}
		}
		ExtractOperatorsInternal(*op->children[0], joins);
		return;
	}
	case LogicalOperatorType::LOGICAL_UNION:
	case LogicalOperatorType::LOGICAL_EXCEPT:
	case LogicalOperatorType::LOGICAL_INTERSECT:
		// AddTableOperator(op);
		ExtractOperatorsInternal(*op->children[0], joins);
		ExtractOperatorsInternal(*op->children[1], joins);
		return;
	case LogicalOperatorType::LOGICAL_WINDOW:
	case LogicalOperatorType::LOGICAL_DUMMY_SCAN:
	case LogicalOperatorType::LOGICAL_EXPRESSION_GET:
	case LogicalOperatorType::LOGICAL_GET:
	case LogicalOperatorType::LOGICAL_EMPTY_RESULT:
	case LogicalOperatorType::LOGICAL_CHUNK_GET:
		AddTableOperator(op);
		return;
	default:
		for (auto &child : op->children) {
			ExtractOperatorsInternal(*child, joins);
		}
	}
}

} // namespace duckdb
