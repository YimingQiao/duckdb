//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/optimizer/thread_scheduler.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include <iostream>
#include <unordered_map>
#include <vector>

namespace duckdb {

using std::string;

class ThreadScheduler {
public:
	static ThreadScheduler &Get() {
		static ThreadScheduler INSTANCE;
		INSTANCE.reservoir_scheduled = 0;
		return INSTANCE;
	}

	// setter
	void SetThreadSetting(size_t value, const vector<string> &sources, const vector<string> &sinks) {
		SetThreadSetting(value, sources, sinks, false);
		SetThreadSetting(value, sources, sinks, true);
	}

	void SetThreadSetting(size_t value, const vector<string> &sources, const vector<string> &sinks, bool has_operator) {
		for (auto &sink : sinks) {
			for (auto &source : sources) {
				SetThreadSetting(value, source, sink, has_operator);
			}
		}
	}

	void SetThreadSetting(size_t value, const string &source, const string &sink) {
		SetThreadSetting(value, source, sink, false);
		SetThreadSetting(value, source, sink, true);
	}

	void SetThreadSetting(size_t value, const string &source, const string &sink, bool has_operator) {
		string key = GenerateKey(source, sink, has_operator);
		thread_setting_[key] = value;
	}

	// getter
	size_t GetThreadSetting(const string &source, const string &sink, bool has_operator) {
		string key = GenerateKey(source, sink, has_operator);
		if (thread_setting_[key] != 0) {
			return thread_setting_[key];
		}

		key = GenerateKey("", sink, has_operator);
		if (thread_setting_[key] != 0) {
			return thread_setting_[key];
		}

		key = GenerateKey(source, "", has_operator);
		if (thread_setting_[key] != 0) {
			return thread_setting_[key];
		}

		// not found!
		return 0;
	}

	void PrintThreadSetting() {
		for (auto &it : thread_setting_) {
			std::cout << it.first << " : " << it.second << "\n";
		}
	}

	inline string GenerateKey(const string &source, const string &sink, bool has_operator) {
		if (has_operator) {
			return source + " -> ... -> " + sink;
		} else {
			return source + " -> " + sink;
		}
	}

private:
	unordered_map<string, size_t> thread_setting_;

public:
	void ResetReservoir() {
		reservoir_scheduled = 0;
	}

	void SetReservoir() {
		++reservoir_scheduled;
	}

	bool GetReservoir() {
		return reservoir_scheduled = 1;
	}

private:
	atomic<int> reservoir_scheduled;
};
} // namespace duckdb