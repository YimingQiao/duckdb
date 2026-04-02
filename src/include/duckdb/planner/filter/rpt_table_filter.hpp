//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/planner/filter/rpt_table_filter.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/planner/table_filter.hpp"
#include "duckdb/planner/table_filter_state.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/selection_vector.hpp"
#include "duckdb/storage/statistics/base_statistics.hpp"
#include "duckdb/planner/filter/selectivity_optional_filter.hpp"

namespace duckdb {

//! RPTFilter — abstract filter interface for bloom/bitmap filters.
//! Analogous to BloomFilter but virtual since we have multiple implementations.
class RPTFilter {
public:
	RPTFilter() : finalized_(false) {
	}
	virtual ~RPTFilter() = default;

	virtual int Lookup(DataChunk &chunk, const vector<idx_t> &bound_cols, SelectionVector &results,
	                   size_t &result_count) const = 0;
	virtual int Lookup(DataChunk &chunk, const vector<idx_t> &bound_cols, Vector &results,
	                   size_t &result_count) const = 0;
	virtual void Insert(DataChunk &chunk, const vector<idx_t> &bound_cols) = 0;
	virtual size_t Hash() const = 0;

	virtual FilterPropagateResult CheckStatistics(const BaseStatistics &stats) {
		return FilterPropagateResult::NO_PRUNING_POSSIBLE;
	}

	virtual string ToString() const {
		return "RPTFilter";
	}

	bool IsValid() const {
		return finalized_;
	}
	void SetValid() {
		finalized_ = true;
	}

	bool finalized_ = false;
};

//! RPTTableFilter — concrete table filter holding an RPTFilter.
//! Modeled after BFTableFilter. Pushed into table scans and called directly
//! from column_segment.cpp, bypassing the ExpressionFilter framework.
class RPTTableFilter final : public TableFilter {
public:
	static constexpr auto TYPE = TableFilterType::RPT_FILTER;

	RPTTableFilter(shared_ptr<RPTFilter> filter_p, LogicalType key_type_p, float selectivity_threshold_p = 1.0f,
	               idx_t n_vectors_to_check_p = 6)
	    : TableFilter(TYPE), filter_(std::move(filter_p)), key_type_(std::move(key_type_p)),
	      selectivity_threshold_(selectivity_threshold_p), n_vectors_to_check_(n_vectors_to_check_p) {
	}

	LogicalType GetKeyType() const {
		return key_type_;
	}

	idx_t Filter(Vector &keys_v, SelectionVector &sel, idx_t &approved_tuple_count, RPTTableFilterState &state) const;

	FilterPropagateResult CheckStatistics(BaseStatistics &stats) const override;
	string ToString(const string &column_name) const override;
	bool Equals(const TableFilter &other) const override;
	unique_ptr<TableFilter> Copy() const override;
	unique_ptr<Expression> ToExpression(const Expression &column) const override;
	void Serialize(Serializer &serializer) const override;

	//! Wrap this filter in a SelectivityOptionalFilter if threshold < 1.0.
	//! Returns a plain RPTTableFilter otherwise.
	static unique_ptr<TableFilter> MakeOptional(shared_ptr<RPTFilter> filter_p, LogicalType key_type_p,
	                                            float selectivity_threshold = 1.0f, idx_t n_vectors_to_check = 6) {
		auto rpt = make_uniq<RPTTableFilter>(std::move(filter_p), std::move(key_type_p), selectivity_threshold,
		                                     n_vectors_to_check);
		if (selectivity_threshold < 1.0f) {
			return make_uniq<SelectivityOptionalFilter>(std::move(rpt), selectivity_threshold, n_vectors_to_check);
		}
		return std::move(rpt);
	}

private:
	shared_ptr<RPTFilter> filter_;
	LogicalType key_type_;
	float selectivity_threshold_;
	idx_t n_vectors_to_check_;
};

} // namespace duckdb
