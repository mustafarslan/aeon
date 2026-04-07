#include "aeon/benchmark_harness.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <queue>
#include <random>
#include <stdexcept>

namespace aeon::bench {

DatasetView DatasetLoader::load_fvecs(const std::filesystem::path &path) {
  std::ifstream is(path, std::ios::binary);
  if (!is) {
    throw std::runtime_error("Failed to open fvecs file: " + path.string());
  }

  uint32_t dim = 0;
  is.read(reinterpret_cast<char *>(&dim), sizeof(uint32_t));
  if (!is) {
    throw std::runtime_error("Failed to read dimensions from fvecs");
  }

  is.seekg(0, std::ios::end);
  size_t file_size = is.tellg();
  is.seekg(0, std::ios::beg);

  size_t row_bytes = sizeof(uint32_t) + dim * sizeof(float);
  if (file_size % row_bytes != 0) {
    throw std::runtime_error("Invalid fvecs file size");
  }

  uint32_t count = file_size / row_bytes;
  DatasetView view;
  view.dim = dim;
  view.count = count;
  view.data.resize(count * dim);

  for (uint32_t i = 0; i < count; ++i) {
    uint32_t cur_dim;
    is.read(reinterpret_cast<char *>(&cur_dim), sizeof(uint32_t));
    if (cur_dim != dim) {
      throw std::runtime_error("Inconsistent dimensions in fvecs");
    }
    is.read(reinterpret_cast<char *>(&view.data[i * dim]), dim * sizeof(float));
  }

  return view;
}

DatasetView DatasetLoader::generate_clustered(uint32_t count, uint32_t dim,
                                              uint32_t clusters, int seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> cluster_dist(-10.0f, 10.0f);
  std::normal_distribution<float> noise_dist(0.0f, 1.0f);

  std::vector<std::vector<float>> cluster_centers(clusters,
                                                  std::vector<float>(dim));
  for (uint32_t c = 0; c < clusters; ++c) {
    for (uint32_t d = 0; d < dim; d++) {
      cluster_centers[c][d] = cluster_dist(rng);
    }
  }

  DatasetView view;
  view.dim = dim;
  view.count = count;
  view.data.resize(count * dim);

  std::uniform_int_distribution<uint32_t> pick_cluster(0, clusters - 1);
  for (uint32_t i = 0; i < count; ++i) {
    uint32_t c = pick_cluster(rng);
    for (uint32_t d = 0; d < dim; ++d) {
      // Create anisotropy by scaling dimensions differently
      float scale = 1.0f + (d % 10);
      view.data[i * dim + d] = cluster_centers[c][d] + noise_dist(rng) * scale;
    }
  }
  return view;
}

std::vector<std::vector<uint64_t>>
GroundTruthCalculator::compute(const DatasetView &db,
                               const DatasetView &queries, uint32_t top_k,
                               simd::MetricType metric) {
  std::vector<std::vector<uint64_t>> ground_truth(queries.count,
                                                  std::vector<uint64_t>(top_k));

  for (uint32_t q = 0; q < queries.count; ++q) {
    auto query = queries.get_vector(q);

    // priority queue to keep track of top_k
    // pair is (distance/similarity, index).
    // For InnerProduct and Cosine, higher is better. For L2, lower is better.
    // For FP64 evaluation, we recalculate purely.

    struct ResultPoint {
      double score;
      uint64_t id;
      bool operator<(const ResultPoint &other) const {
        return score > other.score; // min-heap by default
      }
    };

    auto cmp_l2 = [](const ResultPoint &left, const ResultPoint &right) {
      return left.score < right.score;
    }; // max-heap for L2

    std::priority_queue<ResultPoint> pq_sim;
    std::priority_queue<ResultPoint, std::vector<ResultPoint>, decltype(cmp_l2)>
        pq_diff(cmp_l2);

    for (uint32_t i = 0; i < db.count; ++i) {
      auto point = db.get_vector(i);
      double score = 0.0;

      if (metric == simd::MetricType::L2) {
        for (uint32_t d = 0; d < db.dim; ++d) {
          double diff =
              static_cast<double>(query[d]) - static_cast<double>(point[d]);
          score += diff * diff;
        }
        pq_diff.push({score, static_cast<uint64_t>(i)});
        if (pq_diff.size() > top_k)
          pq_diff.pop();
      } else {
        // InnerProduct or Cosine. We just do exact inner product, assuming
        // Cosine means vectors are normalized or we evaluate just inner product
        // for simplicity in exact GT.
        for (uint32_t d = 0; d < db.dim; ++d) {
          score +=
              static_cast<double>(query[d]) * static_cast<double>(point[d]);
        }
        pq_sim.push({score, static_cast<uint64_t>(i)});
        if (pq_sim.size() > top_k)
          pq_sim.pop();
      }
    }

    std::vector<uint64_t> results(top_k);
    if (metric == simd::MetricType::L2) {
      for (int32_t k = top_k - 1; k >= 0; --k) {
        results[k] = pq_diff.top().id;
        pq_diff.pop();
      }
    } else {
      for (int32_t k = top_k - 1; k >= 0; --k) {
        results[k] = pq_sim.top().id;
        pq_sim.pop();
      }
    }
    ground_truth[q] = results;
  }

  return ground_truth;
}

float RecallCalculator::recall_at_k(
    const std::vector<std::vector<uint64_t>> &predicted,
    const std::vector<std::vector<uint64_t>> &ground_truth, uint32_t k) {
  if (predicted.empty() || ground_truth.empty())
    return 0.0f;

  size_t total_queries = predicted.size();
  size_t total_hits = 0;

  for (size_t q = 0; q < total_queries; ++q) {
    const auto &pred = predicted[q];
    const auto &gt = ground_truth[q];

    uint32_t check_k = std::min(k, static_cast<uint32_t>(gt.size()));
    check_k = std::min(check_k, static_cast<uint32_t>(pred.size()));

    size_t hits = 0;
    for (uint32_t i = 0; i < check_k; ++i) {
      if (std::find(gt.begin(), gt.begin() + check_k, pred[i]) !=
          gt.begin() + check_k) {
        hits++;
      }
    }
    total_hits += hits;
  }

  return static_cast<float>(total_hits) / static_cast<float>(total_queries * k);
}

void ParetoRecorder::record(float latency_ns, float recall,
                            const std::string &kernel_label) {
  points_.push_back({latency_ns, recall, kernel_label});
}

void ParetoRecorder::write_csv(const std::filesystem::path &path) const {
  std::ofstream os(path);
  if (!os) {
    throw std::runtime_error("Failed to open output CSV: " + path.string());
  }
  os << "LatencyNs,Recall,Kernel\n";
  for (const auto &p : points_) {
    os << p.latency << "," << p.recall << "," << p.tag << "\n";
  }
}

} // namespace aeon::bench
