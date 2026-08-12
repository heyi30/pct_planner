#include "sparse_a_star/sparse_a_star_search.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_set>

namespace {

const std::vector<Eigen::Vector2i> kNeighbor2D = {
    Eigen::Vector2i(-1, -1), Eigen::Vector2i(-1, 0), Eigen::Vector2i(-1, 1),
    Eigen::Vector2i(0, -1),  Eigen::Vector2i(0, 1),  Eigen::Vector2i(1, -1),
    Eigen::Vector2i(1, 0),   Eigen::Vector2i(1, 1),
};

// 3x3 neighborhood including the center cell.
const std::vector<Eigen::Vector2i> kNeighbor2DWithCenter = {
    Eigen::Vector2i(-1, -1), Eigen::Vector2i(-1, 0), Eigen::Vector2i(-1, 1),
    Eigen::Vector2i(0, -1),  Eigen::Vector2i(0, 0),  Eigen::Vector2i(0, 1),
    Eigen::Vector2i(1, -1),  Eigen::Vector2i(1, 0),  Eigen::Vector2i(1, 1),
};

}  // namespace

SparseNode* SparseAstar::FindCrossLayerTarget(SparseNode* node, int target_layer) {
  if (target_layer < 0 || target_layer >= max_layers_) {
    return nullptr;
  }
  const int row = node->idx[1];
  const int col = node->idx[2];

  SparseNode* best = nullptr;
  int best_dist_sq = std::numeric_limits<int>::max();
  double best_cost = std::numeric_limits<double>::max();

  for (const auto& d : kNeighbor2DWithCenter) {
    const int nr = row + d[0];
    const int nc = col + d[1];
    if (nr < 0 || nr >= max_y_ || nc < 0 || nc >= max_x_) {
      continue;
    }
    SparseNode* candidate = GetNode(Eigen::Vector3i(target_layer, nr, nc));
    if (candidate == nullptr) {
      continue;
    }
    if (candidate->cost > cost_threshold_ && candidate->gateway == 0) {
      continue;
    }
    if (std::abs(candidate->height - node->height) > step_max_) {
      continue;
    }
    const int dist_sq = d[0] * d[0] + d[1] * d[1];
    if (dist_sq < best_dist_sq || (dist_sq == best_dist_sq && candidate->cost < best_cost)) {
      best_dist_sq = dist_sq;
      best_cost = candidate->cost;
      best = candidate;
    }
  }
  return best;
}

void SparseAstar::Init(const std::array<int, 3>& shape,
                       double resolution,
                       double cost_threshold,
                       double step_max,
                       double cost_weight,
                       const Eigen::MatrixXi& indices,
                       const Eigen::VectorXd& trav,
                       const Eigen::VectorXd& elev_g,
                       const Eigen::VectorXd& elev_c,
                       const Eigen::VectorXi& gateway) {
  auto t0 = std::chrono::high_resolution_clock::now();

  max_layers_ = shape[0];
  max_y_ = shape[1];
  max_x_ = shape[2];
  resolution_ = resolution;
  cost_threshold_ = cost_threshold;
  step_max_ = step_max;
  cost_weight_ = cost_weight;

  nodes_.clear();
  nodes_by_id_.clear();
  result_.clear();
  visited_.clear();

  const int n = indices.rows();
  nodes_.reserve(n * 2 + 1);
  nodes_by_id_.resize(n);

  for (int i = 0; i < n; ++i) {
    auto node = std::make_unique<SparseNode>();
    node->idx = indices.row(i);
    node->node_id = i;
    node->cost = trav(i);
    node->height = elev_g(i);
    node->ceiling = elev_c(i);
    node->gateway = gateway(i);
    nodes_[Hash(node->idx)] = std::move(node);
  }

  for (auto& kv : nodes_) {
    nodes_by_id_[kv.second->node_id] = kv.second.get();
  }

  // Precompute cross-layer targets for gateway nodes.
  for (auto& kv : nodes_) {
    SparseNode* node = kv.second.get();
    if (node->gateway > 0) {
      node->up_target = FindCrossLayerTarget(node, node->idx[0] + 1);
    } else if (node->gateway < 0) {
      node->down_target = FindCrossLayerTarget(node, node->idx[0] - 1);
    }
  }

  auto dt = std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::high_resolution_clock::now() - t0);
  std::cout << "sparse_astar_init_ms = " << dt.count() / 1000.0 << std::endl;
  std::cout << "sparse_astar_nodes = " << n << std::endl;
  std::cout << "sparse_astar_shape = [" << max_layers_ << ", " << max_y_ << ", "
            << max_x_ << "]" << std::endl;
}

void SparseAstar::Reset() {
  for (auto& kv : nodes_) {
    kv.second->g = 1e9;
    kv.second->f = 1e9;
    kv.second->parent = nullptr;
  }
  result_.clear();
  visited_.clear();
}

int SparseAstar::Hash(const Eigen::Vector3i& idx) const {
  return idx[0] * max_y_ * max_x_ + idx[1] * max_x_ + idx[2];
}

SparseNode* SparseAstar::GetNode(const Eigen::Vector3i& idx) {
  auto it = nodes_.find(Hash(idx));
  if (it == nodes_.end()) {
    return nullptr;
  }
  return it->second.get();
}

const SparseNode* SparseAstar::GetNode(const Eigen::Vector3i& idx) const {
  auto it = nodes_.find(Hash(idx));
  if (it == nodes_.end()) {
    return nullptr;
  }
  return it->second.get();
}

std::vector<SparseNode*> SparseAstar::GetNeighbors(SparseNode* node) {
  std::vector<SparseNode*> neighbors;
  neighbors.reserve(8 + 2);

  const int layer = node->idx[0];
  const int row = node->idx[1];
  const int col = node->idx[2];

  // Same-layer 8-neighborhood.
  for (const auto& d : kNeighbor2D) {
    const int nr = row + d[0];
    const int nc = col + d[1];
    if (nr < 0 || nr >= max_y_ || nc < 0 || nc >= max_x_) {
      continue;
    }
    Eigen::Vector3i nidx(layer, nr, nc);
    SparseNode* neighbor = GetNode(nidx);
    if (neighbor == nullptr) {
      continue;
    }
    if (neighbor->cost > cost_threshold_ && neighbor->gateway == 0) {
      continue;
    }
    if (std::abs(neighbor->height - node->height) > step_max_) {
      continue;
    }
    neighbors.push_back(neighbor);
  }

  // Cross-layer movement through precomputed gateway targets.
  if (node->up_target != nullptr) {
    neighbors.push_back(node->up_target);
  }
  if (node->down_target != nullptr) {
    neighbors.push_back(node->down_target);
  }

  return neighbors;
}

double SparseAstar::PathLengthMeters(const Eigen::MatrixXi& path) const {
  if (path.rows() < 2) {
    return 0.0;
  }
  double len = 0.0;
  for (int i = 1; i < path.rows(); ++i) {
    const Eigen::Vector3i a = path.row(i - 1);
    const Eigen::Vector3i b = path.row(i);
    const double dx = (b[2] - a[2]) * resolution_;
    const double dy = (b[1] - a[1]) * resolution_;
    const double dh = GetNode(b)->height - GetNode(a)->height;
    len += std::sqrt(dx * dx + dy * dy + dh * dh);
  }
  return len;
}

double SparseAstar::GetHeuristic(const SparseNode* a, const SparseNode* b) const {
  const Eigen::Vector3i d = a->idx - b->idx;
  const int dx = std::abs(d[0]);
  const int dy = std::abs(d[1]);
  const int dz = std::abs(d[2]);

  if (h_type_ == SparseHeuristicType::kEuclidean) {
    return std::sqrt(dx * dx + dy * dy + dz * dz);
  }
  if (h_type_ == SparseHeuristicType::kManhattan) {
    return dx + dy + dz;
  }
  // Diagonal / octile distance in grid units.
  const int dmin = std::min({dx, dy, dz});
  const int dmax = std::max({dx, dy, dz});
  const int dmid = dx + dy + dz - dmin - dmax;
  return std::sqrt(3.0) * dmin + std::sqrt(2.0) * (dmid - dmin) + (dmax - dmid);
}

bool SparseAstar::Search(const Eigen::Vector3i& start, const Eigen::Vector3i& goal) {
  auto t0 = std::chrono::high_resolution_clock::now();

  if (!result_.empty()) {
    Reset();
  }

  SparseNode* start_node = GetNode(start);
  SparseNode* goal_node = GetNode(goal);
  if (start_node == nullptr) {
    std::cout << "SparseAstar: start not in sparse node set" << std::endl;
    return false;
  }
  if (goal_node == nullptr) {
    std::cout << "SparseAstar: goal not in sparse node set" << std::endl;
    return false;
  }

  start_node->g = 0.0;
  start_node->f = GetHeuristic(start_node, goal_node);

  std::priority_queue<SparseNode*, std::vector<SparseNode*>, SparseNodeCompare> open_set;
  std::unordered_set<int> closed_set;
  open_set.push(start_node);

  while (!open_set.empty()) {
    SparseNode* current = open_set.top();
    open_set.pop();

    const int current_hash = Hash(current->idx);
    if (closed_set.find(current_hash) != closed_set.end()) {
      continue;
    }
    closed_set.insert(current_hash);

    if (current->idx == goal_node->idx) {
      // Reconstruct path.
      while (current != nullptr) {
        result_.push_back(current);
        current = current->parent;
      }
      std::reverse(result_.begin(), result_.end());

      auto dt = std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::high_resolution_clock::now() - t0);
      std::cout << "sparse_astar_search_ms = " << dt.count() / 1000.0 << std::endl;
      std::cout << "visited_nodes = " << visited_.size() << std::endl;
      std::cout << "path_nodes = " << result_.size() << std::endl;
      return true;
    }

    visited_.push_back(current);

    for (SparseNode* neighbor : GetNeighbors(current)) {
      const double dz = (neighbor->height - current->height) / resolution_;
      const Eigen::Vector3i diff = neighbor->idx - current->idx;
      const double dist = std::sqrt(
          diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2] + dz * dz);
      const double cost_penalty = cost_weight_ * neighbor->cost;
      const double tentative_g = current->g + dist + cost_penalty;

      if (tentative_g < neighbor->g) {
        neighbor->g = tentative_g;
        neighbor->f = tentative_g + GetHeuristic(neighbor, goal_node);
        neighbor->parent = current;
        open_set.push(neighbor);
      }
    }
  }

  auto dt = std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::high_resolution_clock::now() - t0);
  std::cout << "sparse_astar_search_ms = " << dt.count() / 1000.0 << std::endl;
  std::cout << "visited_nodes = " << visited_.size() << std::endl;
  std::cout << "path_nodes = 0" << std::endl;
  return false;
}

Eigen::MatrixXi SparseAstar::GetResultMatrix() const {
  if (result_.empty()) {
    return Eigen::MatrixXi();
  }
  Eigen::MatrixXi mat(result_.size(), 3);
  for (size_t i = 0; i < result_.size(); ++i) {
    mat.row(i) = result_[i]->idx;
  }
  return mat;
}

Eigen::MatrixXi SparseAstar::GetVisitedSet() const {
  if (visited_.empty()) {
    return Eigen::MatrixXi();
  }
  Eigen::MatrixXi mat(visited_.size(), 3);
  for (size_t i = 0; i < visited_.size(); ++i) {
    mat.row(i) = visited_[i]->idx;
  }
  return mat;
}

Eigen::MatrixXi SparseAstar::SearchOne(Eigen::Vector3i start, Eigen::Vector3i goal,
                                       SearchScratch& s) {
  SparseNode* start_node = GetNode(start);
  SparseNode* goal_node = GetNode(goal);
  if (start_node == nullptr || goal_node == nullptr) {
    return Eigen::MatrixXi();
  }

  s.Reset();
  const int start_id = start_node->node_id;
  const int goal_id = goal_node->node_id;
  (void)goal_id;

  s.g[start_id] = 0.0;
  s.touched.push_back(start_id);

  // Min-heap of (f, node_id).
  typedef std::pair<double, int> Entry;
  std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> open;
  open.push(Entry(GetHeuristic(start_node, goal_node), start_id));

  // Check the cancel flag every 2048 pops so a long (e.g. failed) search can
  // be aborted by Ctrl+C within a bounded number of expansions.
  constexpr int kCancelCheckEvery = 2048;
  int expansions = 0;
  while (!open.empty()) {
    if ((expansions++ & (kCancelCheckEvery - 1)) == 0 &&
        cancel_requested_.load()) {
      return Eigen::MatrixXi();
    }
    const int id = open.top().second;
    open.pop();
    if (s.settled[id]) {
      continue;
    }
    SparseNode* current = nodes_by_id_[id];
    s.settled[id] = 1;

    if (current == goal_node) {
      // Reconstruct path via parent ids (mirrors single-search behavior).
      std::vector<int> path_ids;
      for (int cur = id; cur != -1; cur = s.parent[cur]) {
        path_ids.push_back(cur);
      }
      std::reverse(path_ids.begin(), path_ids.end());
      Eigen::MatrixXi mat(static_cast<int>(path_ids.size()), 3);
      for (size_t i = 0; i < path_ids.size(); ++i) {
        mat.row(i) = nodes_by_id_[path_ids[i]]->idx;
      }
      return mat;
    }

    const double cur_g = s.g[id];
    for (SparseNode* nb : GetNeighbors(current)) {
      const double dz = (nb->height - current->height) / resolution_;
      const Eigen::Vector3i diff = nb->idx - current->idx;
      const double dist = std::sqrt(
          diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2] + dz * dz);
      const double tentative = cur_g + dist + cost_weight_ * nb->cost;
      const int nid = nb->node_id;
      if (tentative < s.g[nid]) {
        if (s.g[nid] == kSparseInf) {
          s.touched.push_back(nid);
        }
        s.g[nid] = tentative;
        s.parent[nid] = id;
        open.push(Entry(tentative + GetHeuristic(nb, goal_node), nid));
      }
    }
  }

  return Eigen::MatrixXi();
}

std::vector<Eigen::MatrixXi> SparseAstar::SearchBatch(const Eigen::MatrixXi& starts,
                                                      const Eigen::MatrixXi& goals,
                                                      int num_threads,
                                                      bool (*interrupt)(),
                                                      void (*progress)(int, int, int, double)) {
  const int k = starts.rows();
  std::vector<Eigen::MatrixXi> results(k);
  if (k == 0) {
    return results;
  }

  const int n_nodes = static_cast<int>(nodes_by_id_.size());
  if (num_threads <= 0) {
    num_threads = static_cast<int>(std::thread::hardware_concurrency());
  }
  num_threads = std::min(num_threads, k);

  cancel_requested_.store(false);

  if (num_threads <= 1) {
    SearchScratch scratch(n_nodes);
    for (int i = 0; i < k && !cancel_requested_.load(); ++i) {
      if (interrupt && interrupt()) {
        cancel_requested_.store(true);
        break;
      }
      results[i] = SearchOne(starts.row(i), goals.row(i), scratch);
      if (progress) {
        progress(i, k, results[i].rows(), PathLengthMeters(results[i]));
      }
    }
    return results;
  }

  std::vector<SearchScratch> scratch(num_threads, SearchScratch(n_nodes));
  std::atomic<int> active{num_threads};
  std::mutex progress_mutex;
  std::vector<std::thread> threads;
  threads.reserve(num_threads);
  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back([this, &results, &starts, &goals, &scratch, &active,
                          &progress_mutex, k, num_threads, t, progress]() {
      // Round-robin so expensive pairs (e.g. long failed searches) do not all
      // land on the same thread and block a whole contiguous chunk.
      for (int i = t; i < k && !cancel_requested_.load(); i += num_threads) {
        results[i] = SearchOne(starts.row(i), goals.row(i), scratch[t]);
        if (progress) {
          const int n = results[i].rows();
          const double len = PathLengthMeters(results[i]);
          std::lock_guard<std::mutex> lk(progress_mutex);
          progress(i, k, n, len);
        }
      }
      active.fetch_sub(1);
    });
  }

  // Wait, polling the interrupt callback. When it fires, cancel: workers stop
  // at their next pair/search and wind down within ~one expansion chunk. The
  // callback runs on the calling thread with the Python GIL re-acquired, which
  // is what lets a Ctrl+C raise inside the interpreter while we wait here.
  while (active.load() > 0) {
    if (interrupt && interrupt()) {
      cancel_requested_.store(true);
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  for (auto& th : threads) {
    th.join();
  }
  return results;
}
