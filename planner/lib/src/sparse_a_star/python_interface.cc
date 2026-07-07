#include "sparse_a_star/sparse_a_star_search.h"

#include "pybind11/eigen.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;

PYBIND11_MODULE(sparse_a_star, m) {
  py::enum_<SparseHeuristicType>(m, "SparseHeuristicType")
      .value("EUCLIDEAN", SparseHeuristicType::kEuclidean)
      .value("MANHATTAN", SparseHeuristicType::kManhattan)
      .value("DIAGONAL", SparseHeuristicType::kDiagonal)
      .export_values();

  py::class_<SparseAstar>(m, "SparseAstar")
      .def(py::init<SparseHeuristicType>(),
           py::arg("h_type") = SparseHeuristicType::kDiagonal)
      .def("init", &SparseAstar::Init,
           py::arg("shape"),
           py::arg("resolution"),
           py::arg("cost_threshold"),
           py::arg("step_max"),
           py::arg("cost_weight"),
           py::arg("indices"),
           py::arg("trav"),
           py::arg("elev_g"),
           py::arg("elev_c"),
           py::arg("gateway"))
      .def("search", &SparseAstar::Search)
      .def("get_result_matrix", &SparseAstar::GetResultMatrix)
      .def("get_visited_set", &SparseAstar::GetVisitedSet);
}
