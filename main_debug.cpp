#include "duckdb.hpp"

#include <iostream>
#include <chrono> // For timing

int main() {
	// Open DuckDB database (in-memory for simplicity, change "example.db" for a persistent one)
	std::string path = "../duckdb_benchmark_data/tpch_sf1.duckdb";
	duckdb::DuckDB db(path);
	duckdb::Connection conn(db);

	conn.Query("SET threads = 1;");

	std::string query = R"(
		explain analyze
		        SELECT
				    sum(l_extendedprice) / 7.0 AS avg_yearly
				FROM
				    lineitem,
				    part
				WHERE
				    p_partkey = l_partkey
				    AND p_brand = 'Brand#23'
				    AND p_container = 'MED BOX'
				    AND l_quantity < (
				        SELECT
				            0.2 * avg(l_quantity)
				        FROM
				            lineitem
				        WHERE
				            l_partkey = p_partkey);
				)";

	// Execute query twice to warm up
	conn.Query(query);
	conn.Query(query);

	// Execute last query
	auto start = std::chrono::high_resolution_clock::now();
	auto result = conn.Query(query);
	auto end = std::chrono::high_resolution_clock::now();

	// Calculate elapsed time in milliseconds
	double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

	// Check result and print execution time
	if (result->HasError()) {
		std::cerr << "Query error: " << result->GetError() << "\n";
		return 1;
	} else {
		std::cout << result->ToString() << "\n";
		std::cout << "Execution Time: " << elapsed_ms << " ms\n";
	}

	return 0;
}
