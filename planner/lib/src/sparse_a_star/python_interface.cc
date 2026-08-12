#include "sparse_a_star/sparse_a_star_search.h"

#include <cstdio>

#include "pybind11/eigen.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;

// Polled by SparseAstar::SearchBatch's wait loop on the main thread. Re-acquires
// the Python GIL (released around the batch) and runs Python-level signal
// handlers, so Ctrl+C raises KeyboardInterrupt instead of being deferred until
// the whole batch returns. Returns true once a signal handler raised.
static bool InterruptCheck() {
  py::gil_scoped_acquire acquire;
  return PyErr_CheckSignals() != 0;
}

// Live per-pair progress line, matching the format Planner.py prints for a
// finished batch so the user sees each pair as it completes. num_waypoints == 0
// means the search failed. printf + fflush (not std::cout) so progress is
// flushed immediately even when stdout is piped.
static void PrintProgress(int pair_index, int total, int num_waypoints,
                          double length_meters) {
  if (num_waypoints > 0) {
    std::printf("[%d/%d] waypoints=%d, length=%.3f m\n", pair_index + 1, total,
                num_waypoints, length_meters);
  } else {
    std::printf("[%d/%d] Planning failed for pair %d.\n", pair_index + 1, total,
                pair_index);
  }
  std::fflush(stdout);
}

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
      .def("search_batch",
           [](SparseAstar& self, const Eigen::MatrixXi& starts,
              const Eigen::MatrixXi& goals, int num_threads) {
             // Run the batch without the GIL so InterruptCheck can re-acquire
             // it to process Ctrl+C; SearchBatch polls it while workers run.
             // The inner scope ends, the gil_scoped_release destructor re-acquires
             // the GIL, and then we inspect/propagate the pending exception.
             std::vector<Eigen::MatrixXi> results;
             {
               py::gil_scoped_release release;
               results = self.SearchBatch(starts, goals, num_threads,
                                          &InterruptCheck, &PrintProgress);
             }
             if (PyErr_Occurred()) {
               throw py::error_already_set();
             }
             return results;
           },
           py::arg("starts"),
           py::arg("goals"),
           py::arg("num_threads") = 0)
      .def("get_result_matrix", &SparseAstar::GetResultMatrix)
      .def("get_visited_set", &SparseAstar::GetVisitedSet);
}
