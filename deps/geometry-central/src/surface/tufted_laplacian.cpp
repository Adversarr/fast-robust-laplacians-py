#include "geometrycentral/surface/tufted_laplacian.h"
#include "geometrycentral/surface/edge_length_geometry.h"
#include "geometrycentral/surface/intrinsic_mollification.h"
#include "geometrycentral/surface/simple_idt.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include <chrono>
#include <iostream>

namespace geometrycentral {
namespace surface {

std::tuple<SparseMatrix<double>, SparseMatrix<double>>
buildTuftedLaplacian(SurfaceMesh& mesh, EmbeddedGeometryInterface& geom, double relativeMollificationFactor,
                     bool printTiming) {
  auto t_func_start = std::chrono::steady_clock::now();

  // Create a copy of the mesh / geometry to operate on
  std::unique_ptr<SurfaceMesh> tuftedMesh = mesh.copyToSurfaceMesh();
  geom.requireVertexPositions();
  VertexPositionGeometry tuftedGeom(*tuftedMesh, geom.vertexPositions.reinterpretTo(*tuftedMesh));
  tuftedGeom.requireEdgeLengths();
  EdgeData<double> tuftedEdgeLengths = tuftedGeom.edgeLengths;
  if (printTiming) {
    auto t = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> dt = t - t_func_start;
    std::cout << "[Tufted] copy mesh+geom & edge lengths: " << dt.count() << " ms" << std::endl;
  }

  // Mollify, if requested
  if (relativeMollificationFactor > 0) {
    auto t_mollify_start = std::chrono::steady_clock::now();
    mollifyIntrinsic(*tuftedMesh, tuftedEdgeLengths, relativeMollificationFactor);
    if (printTiming) {
      auto t = std::chrono::steady_clock::now();
      std::chrono::duration<double, std::milli> dt = t - t_mollify_start;
      std::cout << "[Tufted] intrinsic mollification: " << dt.count() << " ms" << std::endl;
    }
  }

  // Build the cover
  auto t_cover_start = std::chrono::steady_clock::now();
  buildIntrinsicTuftedCover(*tuftedMesh, tuftedEdgeLengths, &tuftedGeom);
  if (printTiming) {
    auto t = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> dt = t - t_cover_start;
    std::cout << "[Tufted] build intrinsic tufted cover: " << dt.count() << " ms" << std::endl;
  }

  // Flip to delaunay
  auto t_flip_start = std::chrono::steady_clock::now();
  size_t nFlips = flipToDelaunay(*tuftedMesh, tuftedEdgeLengths);
  if (printTiming) {
    auto t = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> dt = t - t_flip_start;
    std::cout << "[Tufted] flip to Delaunay (" << nFlips << " flips): " << dt.count() << " ms" << std::endl;
  }

  // Build the matrices
  auto t_matrices_start = std::chrono::steady_clock::now();
  EdgeLengthGeometry tuftedIntrinsicGeom(*tuftedMesh, tuftedEdgeLengths);
  tuftedIntrinsicGeom.requireCotanLaplacian();
  tuftedIntrinsicGeom.requireVertexLumpedMassMatrix();
  if (printTiming) {
    auto t = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> dt = t - t_matrices_start;
    std::cout << "[Tufted] build L/M matrices: " << dt.count() << " ms" << std::endl;
  }

  if (printTiming) {
    auto t_end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> dt = t_end - t_func_start;
    std::cout << "[Tufted] total function time: " << dt.count() << " ms" << std::endl;
  }
  return std::make_tuple(0.5 * tuftedIntrinsicGeom.cotanLaplacian, 0.5 * tuftedIntrinsicGeom.vertexLumpedMassMatrix);
}

void buildIntrinsicTuftedCover(SurfaceMesh& mesh, EdgeData<double>& edgeLengths, EmbeddedGeometryInterface* posGeom) {

  if (posGeom) {
    posGeom->requireVertexPositions();
    posGeom->requireFaceNormals();
  }

  // == Transform the connectivity of the input in to that of the tufted cover

  // Create two copies of each input face
  HalfedgeData<Halfedge> otherSheet(mesh);
  FaceData<bool> isFront(mesh, true); // original copy serves as front
  for (Face fFront : mesh.faces()) {
    if (!isFront[fFront]) continue; // only process original faces

    // create the new face, orient it in the opposite direction
    Face fBack = mesh.duplicateFace(fFront);

    // read off the correspondence bewteen the halfedges, before inverting
    Halfedge heF = fFront.halfedge();
    Halfedge heB = fBack.halfedge();
    do {
      otherSheet[heF] = heB;
      otherSheet[heB] = heF;
      heF = heF.next();
      heB = heB.next();
    } while (heF != fFront.halfedge());

    mesh.invertOrientation(fBack);

    isFront[fBack] = false;
  }


  // Around each edge, glue back faces to front along newly created edges
  EdgeData<bool> isOrigEdge(mesh, true);
  for (Edge e : mesh.edges()) {
    if (!isOrigEdge[e]) continue;

    // Gather the original faces incident on the edge (represented by the halfedge along the edge)
    std::vector<Halfedge> edgeFaces;
    edgeFaces.reserve(4); // small reserve to avoid frequent reallocations
    for (Halfedge he : e.adjacentHalfedges()) {
      if (isFront[he.face()]) edgeFaces.push_back(he);
    }
    if (edgeFaces.empty()) continue; // guard against unexpected boundary/degenerate cases

    // If we have normals, use them for an angular sort.
    // Otherwise, just use the "natural" (arbitrary) ordering
    if (posGeom && edgeFaces.size() > 2) {

      Vector3 pTail = posGeom->vertexPositions[e.halfedge().tailVertex()];
      Vector3 pTip = posGeom->vertexPositions[e.halfedge().tipVertex()];
      Vector3 edgeVec = pTip - pTail;
      std::array<Vector3, 2> eBasis = edgeVec.buildTangentBasis();

      auto edgeAngle = [&](const Halfedge& he) {
        Vector3 oppVert = posGeom->vertexPositions[he.next().tipVertex()];
        Vector3 outVec = oppVert - pTail; // normalization is unnecessary for atan2(angle)
        // TODO: if oppVert == pTail, outVec is zero; angle undefined. Previous code also ill-defined.
        return ::std::atan2(dot(std::get<1>(eBasis), outVec), dot(std::get<0>(eBasis), outVec));
      };
      auto sortFunc = [&](const Halfedge& heA, const Halfedge& heB) -> bool { return edgeAngle(heA) > edgeAngle(heB); };

      std::sort(edgeFaces.begin(), edgeFaces.end(), sortFunc);
    }


    // Sequentially connect the faces
    Halfedge currHe = edgeFaces.front();
    if (currHe.orientation()) currHe = otherSheet[currHe];
    for (size_t i = 0; i < edgeFaces.size(); i++) {
      Halfedge nextHe = edgeFaces[(i + 1) % edgeFaces.size()];
      if (currHe.orientation() == nextHe.orientation()) nextHe = otherSheet[nextHe];
      Edge newE = mesh.separateToNewEdge(currHe, nextHe);
      isOrigEdge[newE] = false;
      edgeLengths[newE] = edgeLengths[e];
      currHe = otherSheet[nextHe];
    }
  }

  // sanity checks
  // if (mesh.hasBoundary()) throw std::runtime_error("has boundary");
  // if (!mesh.isEdgeManifold()) throw std::runtime_error("not edge manifold");
  // if (!mesh.isOriented()) throw std::runtime_error("not oriented");
}

} // namespace surface
} // namespace geometrycentral
