#pragma once

#include <Object.h>
#include <PDBVector.h>
#include <LambdaCreationFunctions.h>
#include "GCNNode.h"
#include "GCNEdge.h"
#include "JoinComp.h"
#include "GCNMatrix.h"
#include <math.h>

namespace pdb {

	// the sub namespace
	namespace gcn {

		// This class implements the join between node, edges and node again*/
		class GCNTransformSoftmaxJoin : public JoinComp<GCNTransformSoftmaxJoin, GCNNode, GCNNode, GCNMatrix> {

		public:

			ENABLE_DEEP_COPY

			GCNTransformSoftmaxJoin() = default;

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
					for (int J =0; J<columns; J++){
						for (int K = 0; K<rows; K++){
							outData[J] += feature[K] * matrix[K*columns+J];
						}
					}

					// Implement softmax function.
					// Comment out Softmax here.
					// It will get an integer overflow here.


					float sum = 0;
					for (int i = 0; i < columns; i++){
						sum += exp(outData[i]);
					}

					for (int i = 0; i < columns; i++){
						outData[i] = exp(outData[i]) / sum;
					}


					return out;
				});
			}
		};
	}
}

