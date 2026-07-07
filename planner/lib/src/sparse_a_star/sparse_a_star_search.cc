#include "sparse_a_star/sparse_a_star_search.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <queue>
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
  result_.clear();
  visited_.clear();

  const int n = indices.rows();
  nodes_.reserve(n * 2 + 1);

  for (int i = 0; i < n; ++i) {
    auto node = std::make_unique<SparseNode>();
    node->idx = indices.row(i);
    node->cost = trav(i);
    node->height = elev_g(i);
    node->ceiling = elev_c(i);
    node->gateway = gateway(i);
    nodes_[Hash(node->idx)] = std::move(node);
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
