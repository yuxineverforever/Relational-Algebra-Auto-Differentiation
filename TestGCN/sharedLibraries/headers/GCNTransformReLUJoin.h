#pragma once

#include <Object.h>
#include <PDBVector.h>
#include <LambdaCreationFunctions.h>
#include "GCNNode.h"
#include "GCNEdge.h"
#include "JoinComp.h"
#include "GCNMatrix.h"

namespace pdb {

    // the sub namespace
    namespace gcn {

        // This class implements the join between node, edges and node again*/
        class GCNTransformReLUJoin : public JoinComp<GCNTransformReLUJoin, GCNNode, GCNNode, GCNMatrix> {

        public:

            ENABLE_DEEP_COPY

            GCNTransformReLUJoin() = default;

            /* Join condition */
            Lambda<bool> getSelection(Handle<GCNNode> node, Handle<GCNMatrix> transformMatrix) {
                return makeLambda(node, transformMatrix, [&](Handle<GCNNode> &node, Handle<GCNMatrix> &transformMatrix) {
                    return true;
                });
            }

            Lambda<Handle<GCNNode>> getProjection(Handle<GCNNode> node, Handle<GCNMatrix> transformMatrix) {
                return makeLambda(node, transformMatrix, [&](Handle<GCNNode> &node, Handle<GCNMatrix> &transformMatrix) {

                    auto nodeID = node->nodeID;

                    auto columns = transformMatrix->numCols;
                    auto rows = transformMatrix->numRows;

                    Handle<GCNNode> out = makeObject<GCNNode>(nodeID, columns);

                    float * feature = node->data->c_ptr();
                    float * matrix = transformMatrix->data->c_ptr();
                    float * outData = out->data->c_ptr();

                    // Implement a vector matrix product.
                    // already replaced by MKL
                    for (int J =0; J<columns; J++){
                        for (int K = 0; K<rows; K++){
                            outData[J] += feature[K] * matrix[K*columns+J];
                        }
                    }

                    // Implement RELU function.
                    // already replaced by MKL
                    for(int i = 0; i < columns; i++) {
                        if(outData[i] < 0.0f) {
                            outData[i] = 0.0f;
                        }
                    }

                    return out;
                });
            }
        };
    }
}

