#include "point_cloud_utilities.h"

#include "geometrycentral/numerical/linear_algebra_utilities.h"
#include "geometrycentral/surface/edge_length_geometry.h"
#include "geometrycentral/surface/intrinsic_mollification.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/simple_polygon_mesh.h"
#include "geometrycentral/surface/surface_mesh.h"
#include "geometrycentral/surface/surface_mesh_factories.h"
#include "geometrycentral/surface/tufted_laplacian.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "Eigen/Dense"
#include <chrono>
#include <iostream>

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

using namespace geometrycentral;
using namespace geometrycentral::surface;

bool g_printTiming = false;

void setParallism(int count) {
  Eigen::setNbThreads(count);
  omp_set_num_threads(count);
}

void setPrintTiming(bool enable) {
  g_printTiming = enable;
}

// For overloaded functions, with C++11 compiler only
template <typename... Args>
using overload_cast_ = pybind11::detail::overload_cast_impl<Args...>;

// Parameters related to unused elements. Maybe expose these as parameters?
double laplacianReplaceVal = 1.0;
double massReplaceVal = -1e-3;


std::tuple<SparseMatrix<double>, SparseMatrix<double>>
buildMeshLaplacian(const DenseMatrix<double>& vMat, const DenseMatrix<size_t>& fMat, double mollifyFactor) {

  // First, load a simple polygon mesh
  SimplePolygonMesh simpleMesh;

  // Copy to std vector representation
  simpleMesh.vertexCoordinates.resize(vMat.rows());
  for (size_t iP = 0; iP < simpleMesh.vertexCoordinates.size(); iP++) {
    simpleMesh.vertexCoordinates[iP] = Vector3{vMat(iP, 0), vMat(iP, 1), vMat(iP, 2)};
  }
  simpleMesh.polygons.resize(fMat.rows());
  for (size_t iF = 0; iF < simpleMesh.polygons.size(); iF++) {
    simpleMesh.polygons[iF] = std::vector<size_t>{fMat(iF, 0), fMat(iF, 1), fMat(iF, 2)};
  }

  // Remove any unused vertices
  std::vector<size_t> oldToNewMap = simpleMesh.stripUnusedVertices();


  // Build the rich mesh data structure
  std::unique_ptr<SurfaceMesh> mesh;
  std::unique_ptr<VertexPositionGeometry> geometry;
  std::tie(mesh, geometry) = makeSurfaceMeshAndGeometry(simpleMesh.polygons, simpleMesh.vertexCoordinates);

  // Do the hard work, calling the geometry-central function
  SparseMatrix<double> L, M;
  std::tie(L, M) = buildTuftedLaplacian(*mesh, *geometry, mollifyFactor, g_printTiming);

  // If necessary, re-index matrices to account for any unreferenced vertices which were skipped.
  // For any unreferenced verts, creates an identity row/col in the Laplacian and
  bool anyUnreferenced = false;
  for (const size_t& ind : oldToNewMap) {
    if (ind == INVALID_IND) anyUnreferenced = true;
  }
  if (anyUnreferenced) {


    // Invert the map
    std::vector<size_t> newToOldMap(simpleMesh.nVertices());
    for (size_t iOld = 0; iOld < oldToNewMap.size(); iOld++) {
      if (oldToNewMap[iOld] != INVALID_IND) {
        newToOldMap[oldToNewMap[iOld]] = iOld;
      }
    }
    size_t N = oldToNewMap.size();

    { // Update the Laplacian

      std::vector<Eigen::Triplet<double>> triplets;

      // Copy entries
      for (int k = 0; k < L.outerSize(); k++) {
        for (typename SparseMatrix<double>::InnerIterator it(L, k); it; ++it) {
          double thisVal = it.value();
          int i = it.row();
          int j = it.col();
          triplets.emplace_back(newToOldMap[i], newToOldMap[j], thisVal);
        }
      }

      // Add diagonal entries for unreferenced
      for (size_t iOld = 0; iOld < oldToNewMap.size(); iOld++) {
        if (oldToNewMap[iOld] == INVALID_IND) {
          triplets.emplace_back(iOld, iOld, laplacianReplaceVal);
        }
      }

      // Update the matrix
      L = SparseMatrix<double>(N, N);
      L.setFromTriplets(triplets.begin(), triplets.end());
    }

    { // Update the mass matrix
      std::vector<Eigen::Triplet<double>> triplets;

      // Copy entries
      double smallestVal = std::numeric_limits<double>::infinity();
      for (int k = 0; k < M.outerSize(); k++) {
        for (typename SparseMatrix<double>::InnerIterator it(M, k); it; ++it) {
          double thisVal = it.value();
          int i = it.row();
          int j = it.col();
          triplets.emplace_back(newToOldMap[i], newToOldMap[j], thisVal);
          smallestVal = std::fmin(smallestVal, std::abs(thisVal));
        }
      }

      // Add diagonal entries for unreferenced
      double newMassVal = massReplaceVal < 0 ? -massReplaceVal * smallestVal : massReplaceVal;
      for (size_t iOld = 0; iOld < oldToNewMap.size(); iOld++) {
        if (oldToNewMap[iOld] == INVALID_IND) {
          triplets.emplace_back(iOld, iOld, newMassVal);
        }
      }

      // Update the matrix
      M = SparseMatrix<double>(N, N);
      M.setFromTriplets(triplets.begin(), triplets.end());
    }
  }


  return std::make_tuple(L, M);
}

std::tuple<SparseMatrix<double>, SparseMatrix<double>> buildPointCloudLaplacian(const DenseMatrix<double>& vMat,
                                                                                double mollifyFactor, size_t nNeigh) {

  SimplePolygonMesh cloudMesh;

  // Copy to std vector representation
  auto t_func_start = std::chrono::steady_clock::now();
  auto t_copy_start = std::chrono::steady_clock::now();
  cloudMesh.vertexCoordinates.resize(vMat.rows());
  for (size_t iP = 0; iP < cloudMesh.vertexCoordinates.size(); iP++) {
    cloudMesh.vertexCoordinates[iP] = Vector3{vMat(iP, 0), vMat(iP, 1), vMat(iP, 2)};
  }
  if (g_printTiming) {
    auto t = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> dt = t - t_copy_start;
    std::cout << "[PointCloud] copy vertices: " << dt.count() << " ms" << std::endl;
  }

  // Generate the local triangulations for the point cloud
  auto t_knn_start = std::chrono::steady_clock::now();
  Neighbors_t neigh = generate_knn(cloudMesh.vertexCoordinates, nNeigh);
  if (g_printTiming) {
    auto t = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> dt = t - t_knn_start;
    std::cout << "[PointCloud] kNN: " << dt.count() << " ms" << std::endl;
  }
  auto t_normals_start = std::chrono::steady_clock::now();
  std::vector<Vector3> normals = generate_normals(cloudMesh.vertexCoordinates, neigh);
  if (g_printTiming) {
    auto t = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> dt = t - t_normals_start;
    std::cout << "[PointCloud] normals: " << dt.count() << " ms" << std::endl;
  }
  auto t_coords_start = std::chrono::steady_clock::now();
  std::vector<std::vector<Vector2>> coords = generate_coords_projection(cloudMesh.vertexCoordinates, normals, neigh);
  if (g_printTiming) {
    auto t = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> dt = t - t_coords_start;
    std::cout << "[PointCloud] coordinate projection: " << dt.count() << " ms" << std::endl;
  }
  auto t_localtri_start = std::chrono::steady_clock::now();
  LocalTriangulationResult localTri = build_delaunay_triangulations(coords, neigh);
  if (g_printTiming) {
    auto t = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> dt = t - t_localtri_start;
    std::cout << "[PointCloud] local triangulations: " << dt.count() << " ms" << std::endl;
  }

  // Take the union of all triangles in all the neighborhoods
  // Precompute total triangle count to reserve capacity and avoid reallocations
  size_t totalTriangles = 0;
  for (size_t iPt = 0; iPt < cloudMesh.vertexCoordinates.size(); iPt++) {
    totalTriangles += localTri.pointTriangles[iPt].size();
  }
  cloudMesh.polygons.reserve(cloudMesh.polygons.size() + totalTriangles);
  std::chrono::steady_clock::time_point t_union_start;
  bool union_started = false;
  for (size_t iPt = 0; iPt < cloudMesh.vertexCoordinates.size(); iPt++) {
    const std::vector<size_t>& thisNeigh = neigh[iPt];
    size_t nNeigh = thisNeigh.size();

    // Accumulate over triangles
    if (g_printTiming && iPt == 0) {
      // Start timing union on first iteration to cover whole loop
      t_union_start = std::chrono::steady_clock::now();
      union_started = true;
    }
    for (const auto& tri : localTri.pointTriangles[iPt]) {
      std::array<size_t, 3> triGlobal = {iPt, thisNeigh[tri[1]], thisNeigh[tri[2]]};
      cloudMesh.polygons.push_back({triGlobal[0], triGlobal[1], triGlobal[2]});
    }
  }
  if (g_printTiming && union_started) {
    auto t = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> dt = t - t_union_start;
    std::cout << "[PointCloud] union triangles: " << dt.count() << " ms" << std::endl;
  }


  // strip unreferenced vertices (can we argue this should never happen? good regardless for robustness.)
  auto t_strip_start = std::chrono::steady_clock::now();
  std::vector<size_t> oldToNewMap = cloudMesh.stripUnusedVertices();
  if (g_printTiming) {
    auto t = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> dt = t - t_strip_start;
    std::cout << "[PointCloud] strip unreferenced vertices: " << dt.count() << " ms" << std::endl;
  }

  std::unique_ptr<SurfaceMesh> mesh;
  std::unique_ptr<VertexPositionGeometry> geometry;
  auto t_make_mesh_start = std::chrono::steady_clock::now();
  std::tie(mesh, geometry) = makeSurfaceMeshAndGeometry(cloudMesh.polygons, cloudMesh.vertexCoordinates);
  if (g_printTiming) {
    auto t = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> dt = t - t_make_mesh_start;
    std::cout << "[PointCloud] make mesh+geometry: " << dt.count() << " ms" << std::endl;
  }

  SparseMatrix<double> L, M;
  auto t_tufted_start = std::chrono::steady_clock::now();
  std::tie(L, M) = buildTuftedLaplacian(*mesh, *geometry, mollifyFactor, g_printTiming);
  if (g_printTiming) {
    auto t = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> dt = t - t_tufted_start;
    std::cout << "[PointCloud] tufted laplacian total: " << dt.count() << " ms" << std::endl;
  }

  // In-place scaling to avoid allocating new sparse matrices
  L *= (1.0 / 3.0);
  M *= (1.0 / 3.0);
  if (g_printTiming) {
    std::cout << "[PointCloud] scale matrices by 1/3" << std::endl;
  }

  // If necessary, re-index matrices to account for any unreferenced vertices which were skipped.
  // For any unreferenced verts, creates an identity row/col in the Laplacian and
  bool anyUnreferenced = false;
  for (const size_t& ind : oldToNewMap) {
    if (ind == INVALID_IND) anyUnreferenced = true;
  }
  if (anyUnreferenced) {
    auto t_reindex_start = std::chrono::steady_clock::now();


    // Invert the map
    std::vector<size_t> newToOldMap(cloudMesh.nVertices());
    for (size_t iOld = 0; iOld < oldToNewMap.size(); iOld++) {
      if (oldToNewMap[iOld] != INVALID_IND) {
        newToOldMap[oldToNewMap[iOld]] = iOld;
      }
    }
    size_t N = oldToNewMap.size();

    { // Update the Laplacian

      std::vector<Eigen::Triplet<double>> triplets;

      // Copy entries
      for (int k = 0; k < L.outerSize(); k++) {
        for (typename SparseMatrix<double>::InnerIterator it(L, k); it; ++it) {
          double thisVal = it.value();
          int i = it.row();
          int j = it.col();
          triplets.emplace_back(newToOldMap[i], newToOldMap[j], thisVal);
        }
      }

      // Add diagonal entries for unreferenced
      for (size_t iOld = 0; iOld < oldToNewMap.size(); iOld++) {
        if (oldToNewMap[iOld] == INVALID_IND) {
          triplets.emplace_back(iOld, iOld, laplacianReplaceVal);
        }
      }

      // Update the matrix
      L = SparseMatrix<double>(N, N);
      L.setFromTriplets(triplets.begin(), triplets.end());
    }

    { // Update the mass matrix
      std::vector<Eigen::Triplet<double>> triplets;

      // Copy entries
      double smallestVal = std::numeric_limits<double>::infinity();
      for (int k = 0; k < M.outerSize(); k++) {
        for (typename SparseMatrix<double>::InnerIterator it(M, k); it; ++it) {
          double thisVal = it.value();
          int i = it.row();
          int j = it.col();
          triplets.emplace_back(newToOldMap[i], newToOldMap[j], thisVal);
          smallestVal = std::fmin(smallestVal, std::abs(thisVal));
        }
      }

      // Add diagonal entries for unreferenced
      double newMassVal = massReplaceVal < 0 ? -massReplaceVal * smallestVal : massReplaceVal;
      for (size_t iOld = 0; iOld < oldToNewMap.size(); iOld++) {
        if (oldToNewMap[iOld] == INVALID_IND) {
          triplets.emplace_back(iOld, iOld, newMassVal);
        }
      }

      // Update the matrix
      M = SparseMatrix<double>(N, N);
      M.setFromTriplets(triplets.begin(), triplets.end());
    }
    if (g_printTiming) {
      auto t = std::chrono::steady_clock::now();
      std::chrono::duration<double, std::milli> dt = t - t_reindex_start;
      std::cout << "[PointCloud] reindex L/M + fill unreferenced: " << dt.count() << " ms" << std::endl;
    }
  }


  if (g_printTiming) {
    auto t_all_end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> dt = t_all_end - t_func_start;
    std::cout << "[PointCloud] total function time: " << dt.count() << " ms" << std::endl;
  }
  return std::make_tuple(L, M);
}

// Actual binding code
// clang-format off
PYBIND11_MODULE(fast_robust_laplacian_bindings, m) {
  m.doc() = "Robust laplacian low-level bindings";
  m.def("setPrintTiming", &setPrintTiming, "set whether to print timing information", py::arg("enable"));

  m.def("buildMeshLaplacian", &buildMeshLaplacian, "build the mesh Laplacian", 
      py::arg("vMat"), py::arg("fMat"), py::arg("mollifyFactor"));
  
  m.def("buildPointCloudLaplacian", &buildPointCloudLaplacian, "build the point cloud Laplacian", 
      py::arg("vMat"), py::arg("mollifyFactor"), py::arg("nNeigh"));
}

// clang-format on
