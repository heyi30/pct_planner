#pragma once

#include <Eigen/Core>
#include <array>
#include <memory>
#include <unordered_map>
#include <vector>

enum class SparseHeuristicType : int { kEuclidean = 0, kManhattan = 1, kDiagonal = 2 };

struct SparseNode {
  Eigen::Vector3i idx = Eigen::Vector3i(0, 0, 0);  // layer, row, col
  double cost = 0.0;
  double height = 0.0;
  double ceiling = 0.0;
  int gateway = 0;

  // A* state
  double g = 1e9;
  double f = 1e9;
  SparseNode* parent = nullptr;

  // Precomputed cross-layer targets (nullptr if none).
  SparseNode* up_target = nullptr;
  SparseNode* down_target = nullptr;
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

 private:
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
  std::vector<SparseNode*> result_;
  std::vector<SparseNode*> visited_;
};
