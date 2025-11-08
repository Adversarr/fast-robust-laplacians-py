#pragma once

#include "geometrycentral/numerical/linear_algebra_utilities.h"
#include "geometrycentral/surface/embedded_geometry_interface.h"
#include "geometrycentral/surface/surface_mesh.h"

namespace geometrycentral {
namespace surface {


// Build cotan Laplacian and lumped mass matrix on the intrinsic tufted cover.
// relativeMollificationFactor > 0 triggers intrinsic mollification of edge lengths.
// If printTiming is true, the routine prints durations for major stages to stdout.
// TODO: Confirm downstream callers outside this repo don't rely on the 3-arg signature; default arg preserves source compatibility.
std::tuple<SparseMatrix<double>, SparseMatrix<double>>
buildTuftedLaplacian(SurfaceMesh& mesh, EmbeddedGeometryInterface& geom,
                     double relativeMollificationFactor = 0., bool printTiming = false);

// Modifies the input mesh and edge lengths to be the tufted cover!
void buildIntrinsicTuftedCover(SurfaceMesh& mesh, EdgeData<double>& edgeLengths,
                               EmbeddedGeometryInterface* posGeom = nullptr);


} // namespace surface
} // namespace geometrycentral
