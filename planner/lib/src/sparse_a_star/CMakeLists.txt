add_library(sparse_a_star_search SHARED
  sparse_a_star_search.cc
)

pybind11_add_module(sparse_a_star
  python_interface.cc
)

target_link_libraries(sparse_a_star PRIVATE sparse_a_star_search)
