add_subdirectory(expression_executor)
add_subdirectory(nested_loop_join)
add_subdirectory(operator)
add_subdirectory(physical_plan)
add_subdirectory(index)
add_subdirectory(sample)

add_library_unity(
  duckdb_execution
  OBJECT
  adaptive_filter.cpp
  aggregate_hashtable.cpp
  base_aggregate_hashtable.cpp
  column_binding_resolver.cpp
  expression_executor.cpp
  expression_executor_state.cpp
  join_hashtable.cpp
  perfect_aggregate_hashtable.cpp
  physical_operator.cpp
  physical_plan_generator.cpp
  radix_partitioned_hashtable.cpp
  transfer_bf_linker.cpp)
set(ALL_OBJECT_FILES
    ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:duckdb_execution>
    PARENT_SCOPE)
