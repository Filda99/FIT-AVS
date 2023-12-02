/**
 * @file    tree_mesh_builder.cpp
 *
 * @author  Filip Jahn <xjahnf00@stud.fit.vutbr.cz>
 *
 * @brief   Parallel Marching Cubes implementation using OpenMP tasks + octree early elimination
 *
 * @date    DATE
 **/

#include <iostream>
#include <math.h>
#include <limits>

#include "tree_mesh_builder.h"

#define CUT_OFF 1   // Value after which we stop reducing the size of the block

TreeMeshBuilder::TreeMeshBuilder(unsigned gridEdgeSize)
    : BaseMeshBuilder(gridEdgeSize, "Octree")
{

}

unsigned TreeMeshBuilder::marchCubes(const ParametricScalarField &field)
{
    unsigned totalTriangles;
    #pragma omp parallel
    #pragma omp single
    totalTriangles = totalTriangles = octreeDecomposition(Vec3_t<float>(0,0,0), field, mGridSize);
    
    return totalTriangles;
}

unsigned TreeMeshBuilder::octreeDecomposition(const Vec3_t<float> &pos, const ParametricScalarField &field, float edgeSize)
{
    unsigned trianglesCnt = 0u;

    // If we can't keep devide cube, call buildCube function
    // and return result of that function
    if (edgeSize <= CUT_OFF) return buildCube(pos, field);
    
   
    // The length of the edge of an eighth of the cube
    float edgeSizeOfChild = edgeSize / 2.0f;            
    // Half the length of the edge of an eighth of the cube to calculate the center of the cube
    float middlePointEdgeSize = edgeSizeOfChild / 2.0f; 
    
    // Divide current block into 8 smaller blocks
    // In the cycle we move along the individual edges of smaller blocks
    for(auto vertexNormPos: sc_vertexNormPos)
    {
        #pragma omp task shared(trianglesCnt)
        {
            // Count points of smaller cube in our cube
            Vec3_t<float> posInField = pos;
            float dx = (vertexNormPos.x * edgeSizeOfChild);
            float dy = (vertexNormPos.y * edgeSizeOfChild);
            float dz = (vertexNormPos.z * edgeSizeOfChild);
            posInField.x += dx;
            posInField.y += dy;
            posInField.z += dz;
            
            // Count equation if we should process the child as follows:
            // 1. Calculate the middle of small cube
            // Using same equation as in transformCubeVertices()
            Vec3_t<float> middlePoint = {
                (posInField.x + middlePointEdgeSize) * mGridResolution, 
                (posInField.y + middlePointEdgeSize) * mGridResolution, 
                (posInField.z + middlePointEdgeSize) * mGridResolution
            };
            
            // 2. Call evaluateFieldAt to get number of points in small cube
            float f_x = evaluateFieldAt(middlePoint, field);

            // 3. Calculate equation and compare to f_x
            // Precompute constants if they are constant within a specific context
            const double sqrt3 = sqrt(3.0f);
            const double half = 0.5f;
            double result = mIsoLevel + (sqrt3 * edgeSizeOfChild * mGridResolution) * half;

            if (f_x <= result)
            {
                #pragma omp atomic update
                trianglesCnt += octreeDecomposition(posInField, field, edgeSizeOfChild);
            }
        }
    }
   

    #pragma omp taskwait
    return trianglesCnt;
}

float TreeMeshBuilder::evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field)
{
    // NOTE: This method is called from "buildCube(...)"!

    // 1. Store pointer to and number of 3D points in the field
    //    (to avoid "data()" and "size()" call in the loop).
    const Vec3_t<float> *pPoints = field.getPoints().data();
    const unsigned count = unsigned(field.getPoints().size());

    float value = std::numeric_limits<float>::max();

    // 2. Find minimum square distance from points "pos" to any point in the field.
    for(unsigned i = 0; i < count; ++i)
    {
        float dx = (pos.x - pPoints[i].x);
        float dy = (pos.y - pPoints[i].y);
        float dz = (pos.z - pPoints[i].z);

        float distanceSquared = dx * dx + dy * dy + dz * dz;
        // Comparing squares instead of real distance to avoid unnecessary
        // "sqrt"s in the loop.
        value = std::min(value, distanceSquared);
    }

    // 3. Finally take square root of the minimal square distance to get the real distance
    return sqrt(value);
}

void TreeMeshBuilder::emitTriangle(const BaseMeshBuilder::Triangle_t &triangle)
{
    #pragma omp critical
    mTriangles.push_back(triangle);
}
