add_library_unity(
  duckdb_operator_reservoir
  OBJECT
  physical_reservoir.cpp)
set(ALL_OBJECT_FILES
    ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:duckdb_operator_reservoir>
    PARENT_SCOPE)
