#pragma once

#include <Eigen/Core>
#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

enum class SparseHeuristicType : int { kEuclidean = 0, kManhattan = 1, kDiagonal = 2 };

constexpr double kSparseInf = 1e9;

struct SparseNode {
  Eigen::Vector3i idx = Eigen::Vector3i(0, 0, 0);  // layer, row, col
  double cost = 0.0;
  double height = 0.0;
  double ceiling = 0.0;
  int gateway = 0;
  int node_id = -1;  // dense 0..N-1, assigned during Init

  // A* state (single-search only; SearchBatch uses SearchScratch instead)
  double g = 1e9;
  double f = 1e9;
  SparseNode* parent = nullptr;

  // Precomputed cross-layer targets (nullptr if none).
  SparseNode* up_target = nullptr;
  SparseNode* down_target = nullptr;

  // Reverse cross-layer links: the nodes whose up/down target is this one. A
  // cross-layer edge is stored on its source only, so without these the graph
  // would be directed and a pair reachable in one direction would be
  // unreachable in the other -- while the tomogram was pruned assuming
  // undirected connectivity, so every node is nominally reachable. Lazily
  // allocated: only the (few) gateway targets ever receive one.
  std::unique_ptr<std::vector<SparseNode*>> back_links;
};

// Per-thread, per-search scratch for SearchBatch. Arrays are indexed by
// node_id so concurrent searches never touch the shared SparseNode state.
// `touched` records every id written this search so the next search can reset
// only those entries (O(visited) instead of O(N)).
struct SearchScratch {
  std::vector<double> g;
  std::vector<int> parent;
  std::vector<uint8_t> settled;
  std::vector<int> touched;

  explicit SearchScratch(int n = 0) { Resize(n); }

  void Resize(int n) {
    g.assign(n, kSparseInf);
    parent.assign(n, -1);
    settled.assign(n, 0);
    touched.clear();
  }

  void Reset() {
    for (int id : touched) {
      g[id] = kSparseInf;
      parent[id] = -1;
      settled[id] = 0;
    }
    touched.clear();
  }
};

struct SparseNodeCompare {
  bool operator()(const SparseNode* a, const SparseNode* b) const {
    return a->f > b->f;
  }
};

class SparseAstar {
 public:
  explicit SparseAstar(SparseHeuristicType h_type = SparseHeuristicType::kDiagonal)
      : h_type_(h_type) {}
  ~SparseAstar() = default;

  void Init(const std::array<int, 3>& shape,
            double resolution,
            double cost_threshold,
            double step_max,
            double cost_weight,
            const Eigen::MatrixXi& indices,
            const Eigen::VectorXd& trav,
            const Eigen::VectorXd& elev_g,
            const Eigen::VectorXd& elev_c,
            const Eigen::VectorXi& gateway);

  void Reset();

  bool Search(const Eigen::Vector3i& start, const Eigen::Vector3i& goal);

  Eigen::MatrixXi GetResultMatrix() const;
  Eigen::MatrixXi GetVisitedSet() const;

  // Plan K start/goal pairs in one call, thread-parallel over pairs. Each
  // thread owns a SearchScratch, so concurrent searches never race on the
  // shared graph. Returns one Mx3 path matrix per pair (0x3 on failure).
  // num_threads <= 0 means std::thread::hardware_concurrency().
  //
  // `interrupt`, when non-null, is polled from the calling thread while the
  // batch runs; when it returns true, an in-flight batch is canceled: workers
  // stop after their current pair and return empty matrices for the rest. It
  // is only meaningful when the calling thread holds no Python GIL (the pybind
  // binding releases it); it must return false when there is no interrupt.
  //
  // `progress`, when non-null, is invoked as each pair finishes (from the
  // thread that searched it, serialized under an internal mutex) with the pair
  // index, total pair count, path waypoint count (0 on failure), and path
  // length in meters. It is a plain C++ call (no GIL involved); the pybind
  // binding uses it to print live progress lines.
  std::vector<Eigen::MatrixXi> SearchBatch(const Eigen::MatrixXi& starts,
                                           const Eigen::MatrixXi& goals,
                                           int num_threads = 0,
                                           bool (*interrupt)() = nullptr,
                                           void (*progress)(int pair_index,
                                                            int total,
                                                            int num_waypoints,
                                                            double length_meters) = nullptr);

  // Request cancellation of any in-flight SearchBatch/SearchOne.
  void RequestCancel() { cancel_requested_.store(true); }
  bool CancelRequested() const { return cancel_requested_.load(); }

 private:
  Eigen::MatrixXi SearchOne(Eigen::Vector3i start, Eigen::Vector3i goal,
                            SearchScratch& scratch);

  // Sum of world-space segment lengths (x/y scaled by resolution_, z from
  // node heights) along a path matrix; 0.0 for fewer than 2 waypoints.
  double PathLengthMeters(const Eigen::MatrixXi& path) const;

  int Hash(const Eigen::Vector3i& idx) const;
  SparseNode* GetNode(const Eigen::Vector3i& idx);
  const SparseNode* GetNode(const Eigen::Vector3i& idx) const;
  SparseNode* FindCrossLayerTarget(SparseNode* node, int target_layer);
  std::vector<SparseNode*> GetNeighbors(SparseNode* node);
  double GetHeuristic(const SparseNode* a, const SparseNode* b) const;

 private:
  SparseHeuristicType h_type_ = SparseHeuristicType::kDiagonal;

  int max_x_ = 0;
  int max_y_ = 0;
  int max_layers_ = 0;
  double resolution_ = 0.0;
  double cost_threshold_ = 35.0;
  double step_max_ = 0.5;
  double cost_weight_ = 0.2;

  std::unordered_map<int, std::unique_ptr<SparseNode>> nodes_;
  std::vector<SparseNode*> nodes_by_id_;  // node_id -> node, filled during Init
  std::vector<SparseNode*> result_;
  std::vector<SparseNode*> visited_;

  // Set by RequestCancel() (or the SearchBatch interrupt callback); workers
  // poll it and stop early, so Ctrl+C can abort a long batch promptly.
  std::atomic<bool> cancel_requested_{false};
};
