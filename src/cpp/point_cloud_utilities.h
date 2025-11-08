#pragma once

#include "geometrycentral/utilities/vector2.h"
#include "geometrycentral/utilities/vector3.h"

#include "geometrycentral/numerical/linear_algebra_utilities.h"

using geometrycentral::SparseMatrix;
using geometrycentral::Vector;
using geometrycentral::Vector2;
using geometrycentral::Vector3;


// === Basic utility methods

using Neighbors_t = std::vector<std::vector<size_t>>;

// Generate the k-nearest-neighbors for the points.
// The list will _not_ include the center point.
// Note: Implementation may parallelize the per-point queries when `USE_OPENMP` is defined.
// TODO: nanoflann KD-tree concurrent query thread-safety is assumed; verify for your build.
Neighbors_t generate_knn(const std::vector<Vector3>& points, size_t k);

// Estimate normals from a neighborhood (arbitrarily oriented)
// Computes per-point SVD over local neighbor offsets; can be parallelized under `USE_OPENMP`.
std::vector<Vector3> generate_normals(const std::vector<Vector3>& points, const Neighbors_t& neigh);

// Project a neighborhood to 2D tangent plane
// The output is in correspondence with `neigh`, with the center point implicitly at (0,0)
std::vector<std::vector<Vector2>> generate_coords_projection(const std::vector<Vector3>& points,
                                                             const std::vector<Vector3>& normals,
                                                             const Neighbors_t& neigh);


struct LocalTriangulationResult {
  // all triangle indices are in to the neighbors list
  std::vector<std::vector<std::array<size_t, 3>>> pointTriangles; // the triangles which touch the center vertex, always
                                                                  // numbered such that the center vertex comes first
};

// Build incident Delaunay-like triangles per point in 2D projected neighborhoods.
// Outer loop may be parallelized under `USE_OPENMP`; algorithmic behavior unchanged.
LocalTriangulationResult build_delaunay_triangulations(const std::vector<std::vector<Vector2>>& coords,
                                                       const Neighbors_t& neigh);
