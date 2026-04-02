#include "duckdb/planner/filter/rpt_table_filter.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"

namespace duckdb {

idx_t RPTTableFilter::Filter(Vector &keys_v, SelectionVector &sel, idx_t &approved_tuple_count,
                             RPTTableFilterState &state) const {
	if (state.current_capacity < approved_tuple_count) {
		state.result_sel.Initialize(approved_tuple_count);
		state.current_capacity = approved_tuple_count;
	}

	// Slice keys by sel, like BFTableFilter::HashInternal
	DataChunk chunk;
	if (sel.IsSet()) {
		state.keys_sliced_v.Slice(keys_v, sel, approved_tuple_count);
		chunk.data.emplace_back(state.keys_sliced_v);
	} else {
		chunk.data.emplace_back(keys_v);
	}
	chunk.SetCardinality(approved_tuple_count);

	size_t result_count = 0;
	filter_->Lookup(chunk, {0}, state.result_sel, result_count);

	// All elements found — no translation needed
	if (result_count == approved_tuple_count) {
		return approved_tuple_count;
	}

	// Map flat result indices back through original sel
	if (sel.IsSet()) {
		for (idx_t i = 0; i < result_count; i++) {
			const idx_t flat_sel_idx = state.result_sel.get_index(i);
			const idx_t original_sel_idx = sel.get_index(flat_sel_idx);
			sel.set_index(i, original_sel_idx);
		}
	} else {
		sel.Initialize(state.result_sel);
	}

	approved_tuple_count = result_count;
	return approved_tuple_count;
}

FilterPropagateResult RPTTableFilter::CheckStatistics(BaseStatistics &stats) const {
	return filter_->CheckStatistics(stats);
}

string RPTTableFilter::ToString(const string &column_name) const {
	return column_name + " IN RPT(" + filter_->ToString() + ")";
}

bool RPTTableFilter::Equals(const TableFilter &other_p) const {
	if (!TableFilter::Equals(other_p)) {
		return false;
	}
	auto &other = other_p.Cast<RPTTableFilter>();
	return filter_.get() == other.filter_.get();
}

unique_ptr<TableFilter> RPTTableFilter::Copy() const {
	return make_uniq<RPTTableFilter>(filter_, key_type_, selectivity_threshold_, n_vectors_to_check_);
}

unique_ptr<Expression> RPTTableFilter::ToExpression(const Expression &column) const {
	auto bound_constant = make_uniq<BoundConstantExpression>(Value(true));
	return std::move(bound_constant);
}

void RPTTableFilter::Serialize(Serializer &serializer) const {
	TableFilter::Serialize(serializer);
}

} // namespace duckdb
