//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/operator/reservoir/function_profiler.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/chrono.hpp"
#include "duckdb/common/profiler.hpp"

#include <algorithm>
#include <iostream>
#include <mutex>

namespace duckdb {

class BeeProfiler {
public:
	const static bool kEnableProfiling = false;

public:
	static BeeProfiler &Get() {
		static BeeProfiler instance;
		return instance;
	}

	void InsertStatRecord(const string &name, double value) {
		InsertStatRecord(name, uint64_t(value * 1e9));
	}

	inline void InsertStatRecord(const string &name, uint64_t value) {
		if (kEnableProfiling) {
			std::lock_guard<std::mutex> lock(mtx);
			values_[name] += value;
			calling_times_[name] += 1;
		}
	}

	void EndProfiling() {
		if (kEnableProfiling) {
			PrintResults();
			Clear();
		}
	}

	void PrintResults() const {
		std::lock_guard<std::mutex> lock(mtx);

		// -------------------------------- Print Timing Results --------------------------------
		std::vector<std::string> keys;
		keys.reserve(values_.size());
		for (const auto &pair : values_) {
			keys.push_back(pair.first);
		}
		if (!keys.empty()) {
			std::sort(keys.begin(), keys.end());
			std::cerr << "-------\n";
			for (const auto &key : keys) {
				double time = values_.at(key) / double(1e9);
				uint64_t calling_times = calling_times_.at(key);
				double avg = time / calling_times;

				std::cerr << "Total: " << time << " s\tCalls: " << calling_times << "\tAvg: " << avg << " s\t" << key
				          << '\n';
			}
		}
	}

	void Clear() {
		values_.clear();
		calling_times_.clear();
	}

private:
	std::unordered_map<string, uint64_t> values_;
	std::unordered_map<string, uint64_t> calling_times_;
	mutable std::mutex mtx;
};
} // namespace duckdb