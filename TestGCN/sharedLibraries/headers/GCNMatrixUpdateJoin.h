
#pragma once

/*
 * This class implements the update for the learnable matrix.
 */
#include <Object.h>
#include <PDBVector.h>
#include <LambdaCreationFunctions.h>
#include "GCNNode.h"
#include "GCNEdge.h"
#include "JoinComp.h"
#include "GCNMatrix.h"

namespace pdb::gcn {

    class GCNMatrixUpdateJoin : public JoinComp<GCNMatrixUpdateJoin,GCNMatrix, GCNMatrix, GCNMatrix>{
    public:

        ENABLE_DEEP_COPY

        GCNMatrixUpdateJoin() = default;

        Lambda<bool> getSelection(Handle<GCNMatrix> gradient, Handle<GCNMatrix> matrix) {
            return makeLambda(gradient, matrix, [](pdb::Handle<GCNMatrix>& gradient, pdb::Handle<GCNMatrix>& matrix) {
                return true;
            });
        }

        Lambda<Handle<GCNMatrix>> getProjection(Handle<GCNMatrix> gradient, Handle<GCNMatrix> matrix) {
            return makeLambda(gradient, matrix, [](Handle<GCNMatrix> &gradient, Handle<GCNMatrix> &matrix) {

                Handle<GCNMatrix> out = makeObject<GCNMatrix>(matrix->numRows, matrix->numCols);

                float learningRate = 0.1;
                float* gradientData = gradient->data->c_ptr();
                float* matrixData = matrix->data->c_ptr();
                float* outData = out->data->c_ptr();

                for (size_t i = 0; i < matrix->numRows*matrix->numCols; i++){
                    outData[i] = matrixData[i] - learningRate * gradientData[i];
                }
                return out;
            });
        }
    };
}

