#include "duckdb.hpp"
#include "duckdb/execution/operator/reservoir/function_profiler.hpp"
#include "duckdb/execution/operator/reservoir/thread_scheduler.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

void GenerateTables(duckdb::Connection &con, int vector_dim);
void LoadTablesFromParquet(duckdb::Connection &con);
bool CheckParquetFilesExist();
std::string GenerateEmbeddingSQL(int dim, int precision = 2);

auto &scheduler = duckdb::ThreadScheduler::Get();
auto &profiler = duckdb::BeeProfiler::Get();

void ExecuteQuery(duckdb::Connection &con, const std::string &query, bool print = true) {
	size_t run_times = 2;

	scheduler.ResetReservoir();
	con.Query(query); // Perform a warm-up run
	//	std::cerr << "------------------------------ Warm Up ------------------------------\n";

	// Measure execution time
	double total_time = 0;
	for (size_t i = 0; i < run_times; ++i) {
		scheduler.ResetReservoir();
		profiler.Clear();

		auto start = std::chrono::high_resolution_clock::now();

		auto result = con.Query(query);

		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = end - start;

		total_time += elapsed.count();
		profiler.PrintResults();

		if (result->HasError()) {
			std::cerr << result->ToString() << " ";
		}

		std::cerr << "------------------------------ Run " << (i + 1) << " Time: " << elapsed.count()
		          << "s ------------------------------\n";
	}

	// Print query plan
	if (print) {
		auto result = con.Query("EXPLAIN ANALYZE " + query);
		std::cerr << "------------------------------ Query Plan ------------------------------\n";
		std::cerr << result->ToString() << " ";
	}

	// Calculate and print average time
	double avg_time = total_time / run_times;
	std::cerr << "<Average Execution Time: " << avg_time << "s>\n";
}

int main(int argc, char *argv[]) {
	// Check if the number of threads is provided as an input argument
	if (argc != 3) {
		throw std::invalid_argument("Usage: <program_name> <number_of_threads> <reservoir_num_threads>");
	}

	// Parse the number of threads from input arguments
	int num_threads = std::stoi(argv[1]);
	int reservoir_num_threads = std::stoi(argv[2]);

	// nullptr means in-memory database.
	duckdb::DuckDB db("");
	duckdb::Connection con(db);

	// ------------------------------------ DuckDB Settings -------------------------------------------------
	// set num of thread, we cannot use 192 threads because 2 threads are left for Perf.
	{
		con.Query("SET threads TO " + std::to_string(num_threads) + ";");
	}

	// set the allocator flush threshold
	{
		con.Query("SET allocator_flush_threshold=\"1gb\"; ");
	}

	// disable the object cache
	{
		con.Query("PRAGMA disable_object_cache;");
	}

	{
		con.Query("SET allocator_background_threads = true");
	}

	// ---------------------------------------- Load Data --------------------------------------------------
	int vector_dim = 128; // dimension of the embedding vector

	if (!CheckParquetFilesExist()) {
		GenerateTables(con, vector_dim);
	}
	LoadTablesFromParquet(con);

	// ------------------------------------------ Query -----------------------------------------------------
	{
		std::string target_embedding = GenerateEmbeddingSQL(vector_dim);

		// std::string query = "SELECT student.stu_id, room.room_id "
		//                     "FROM student, room "
		//                     "WHERE student.stu_id = room.stu_id "
		//                     "AND student.stu_id <= 5e7 "
		// 					"AND room.room_id <= 5e7"
		//                     "AND list_cosine_similarity(student.embedding, " +
		//                     target_embedding + ") > 0.72;";

		// std::string query = "SELECT student.stu_id, room.room_id "
		//                     "FROM student, room, department "
		//                     "WHERE student.stu_id = room.stu_id "
		//                                 "AND department.major_id = student.major_id "
		//                     "AND student.stu_id <= 5e7 "
		//                     "AND list_cosine_similarity(student.embedding, " +
		//                     target_embedding + ") > 0.5;";

		// std::string query = "SELECT student.stu_id, dept_room_type_join.dept_name, dept_room_type_join.room_id, "
		//                     "dept_room_type_join.type "
		//                     "FROM student "
		//                     "INNER JOIN ( "
		//                     "    SELECT department.name AS dept_name, room_type_join.stu_id, room_type_join.room_id,
		//                     " "room_type_join.type " "    FROM department " "    INNER JOIN ( " "        SELECT
		//                     room.stu_id, room.room_id, type.type " "        FROM room " "        INNER JOIN type " "
		//                     ON room.type = type.type " "    ) AS room_type_join " "    ON department.major_id =
		//                     room_type_join.stu_id "
		// 					"    WHERE department.major_id <= 5e6 "
		//                     ") AS dept_room_type_join "
		//                     "ON student.stu_id = dept_room_type_join.stu_id "
		//                     "WHERE list_cosine_similarity(student.embedding, " + target_embedding + ") > 0.72;";

		std::string query = "SELECT result.stu_id, result.age, result.room_id, result.type, "
		                    "FROM ("
		                    "    SELECT s.stu_id, s.age, s.embedding, rt.room_id, rt.type "
		                    "    FROM student s "
		                    "    JOIN ("
		                    "        SELECT room.stu_id, room.room_id, type.type "
		                    "        FROM room "
		                    "        JOIN type ON room.type = type.type "
		                    "    ) AS rt "
		                    "    ON s.stu_id = rt.stu_id"
		                    ") AS result "
		                    "WHERE result.stu_id <= 5e7 "
		                    "AND list_cosine_similarity(result.embedding, " +
		                    target_embedding + ") > 0.7;";

		// std::string query = "SELECT sd.stu_id, sd.name, rt.room_id, rt.type "
		//                     "FROM "
		//                     "    (SELECT student.stu_id, student.major_id, department.major_id, department.name "
		//                     "        FROM student, department "
		//                     "        WHERE student.major_id = department.major_id "
		//                     "		 AND list_cosine_similarity(student.embedding, " + target_embedding + ") > 0.5"
		//                     "    ) AS sd "
		//                     "INNER JOIN "
		//                     "    (SELECT room.stu_id, room.room_id, type.type "
		//                     "        FROM room, type "
		//                     "        WHERE room.type = type.type "
		//                     "		 AND list_cosine_similarity(room.embedding, " + target_embedding + ") > 0.5"
		//                     "    ) AS rt "
		//                     "ON sd.stu_id = rt.stu_id "
		//                     "WHERE sd.stu_id <= 5e7;";

		scheduler.SetThreadSetting(reservoir_num_threads, "TABLE_SCAN", "RESERVOIR");
		std::cerr << "------------------------------ [Threads = " << reservoir_num_threads
		          << "] ------------------------------\n";
		ExecuteQuery(con, query, true);
	}
}

void GenerateTables(duckdb::Connection &con, int vector_dim) {
	// Create the student table
	con.Query("CREATE OR REPLACE TABLE student AS "
	          "SELECT "
	          "    CAST(stu_id AS INT) AS stu_id, "
	          "    CAST((RANDOM() * 5e7) AS INT) AS major_id, "
	          "    CAST((RANDOM() * 100) AS TINYINT) AS age, "
	          "    (SELECT ARRAY_AGG(RANDOM()) FROM generate_series(1, " +
	          std::to_string(vector_dim) +
	          ")) AS embedding "
	          "FROM generate_series(1, CAST(5e7 AS INT)) vals(stu_id);");

	// Create the department table
	con.Query("CREATE OR REPLACE TABLE department AS "
	          "SELECT "
	          "    CAST(major_id * 8 % 5e7 AS INT) AS major_id, "
	          "    major_id AS name "
	          "FROM generate_series(1, CAST(5e7 AS INT)) vals(major_id);");

	// Create the room table
	con.Query("CREATE OR REPLACE TABLE room AS "
	          "SELECT "
	          "    room_id AS room_id, "
	          "    CAST(room_id * 8 % 5e7 AS INT) AS stu_id, "
	          "    CAST((RANDOM() * 5e7) AS INT) AS type, "
	          "    (SELECT ARRAY_AGG(RANDOM()) FROM generate_series(1, " +
	          std::to_string(vector_dim) +
	          ")) AS embedding "
	          "FROM generate_series(1, CAST(5e7 AS INT)) vals(room_id);");

	// Create the type table
	con.Query("CREATE OR REPLACE TABLE type AS "
	          "SELECT "
	          "    CAST(type * 8 % 5e7 AS INT) AS type, "
	          "    'room_type_' || type AS info "
	          "FROM generate_series(1, CAST(5e7 AS INT)) vals(type);");

	// Export tables to Parquet files
	con.Query("COPY student TO 'student.parquet' (FORMAT 'parquet');");
	con.Query("COPY department TO 'department.parquet' (FORMAT 'parquet');");
	con.Query("COPY room TO 'room.parquet' (FORMAT 'parquet');");
	con.Query("COPY type TO 'type.parquet' (FORMAT 'parquet');");

	std::cout << "Tables created and exported to Parquet files successfully!\n";
}

bool FileExists(const std::string &filePath) {
	std::ifstream file(filePath);
	return file.good();
}

bool CheckParquetFilesExist() {
	return FileExists("student.parquet") && FileExists("department.parquet") && FileExists("room.parquet") &&
	       FileExists("type.parquet");
}

void LoadTablesFromParquet(duckdb::Connection &con) {
	// Load the student table from the Parquet file as a temporary table
	con.Query("CREATE TEMPORARY TABLE student AS SELECT * FROM parquet_scan('student.parquet');");
	con.Query("CREATE TEMPORARY TABLE department AS SELECT * FROM parquet_scan('department.parquet');");
	con.Query("CREATE TEMPORARY TABLE room AS SELECT * FROM parquet_scan('room.parquet');");
	con.Query("CREATE TEMPORARY TABLE type AS SELECT * FROM parquet_scan('type.parquet');");

	// Load the student table from the Parquet file as a temporary table
	// con.Query("CREATE VIEW student AS SELECT * FROM parquet_scan('student.parquet');");
	// con.Query("CREATE VIEW department AS SELECT * FROM parquet_scan('department.parquet');");
	// con.Query("CREATE VIEW room AS SELECT * FROM parquet_scan('room.parquet');");
	// con.Query("CREATE VIEW type AS SELECT * FROM parquet_scan('type.parquet');");

	// Log the operation
	std::cerr << "Loaded student, department, room, and type tables from Parquet files as temporary tables.\n";
}

std::string GenerateEmbeddingSQL(int dim, int precision) {
	// Initialize random number generator
	std::random_device rd;
	std::mt19937 gen(42);
	std::uniform_real_distribution<> dis(0.0, 1.0);

	// Generate random embedding vector and format it as a SQL string
	std::ostringstream oss;
	oss << "LIST_VALUE(" << std::fixed << std::setprecision(precision);
	for (int i = 0; i < dim; ++i) {
		oss << dis(gen);
		if (i < dim - 1) {
			oss << ", ";
		}
	}
	oss << ")";
	return oss.str();
}