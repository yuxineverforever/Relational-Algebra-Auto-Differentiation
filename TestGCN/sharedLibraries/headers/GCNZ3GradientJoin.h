#pragma once

#include <Object.h>
#include <PDBVector.h>
#include <LambdaCreationFunctions.h>
#include "GCNNode.h"
#include "GCNEdge.h"
#include "JoinComp.h"
#include "GCNMatrix.h"

namespace pdb::gcn {

	// This class implements the gradients for Z3.
	class GCNZ3GradientJoin : public JoinComp<GCNZ3GradientJoin, GCNNode, GCNNode, GCNMatrix> {

	public:

		ENABLE_DEEP_COPY

		GCNZ3GradientJoin() = default;

		/* Join condition */
		Lambda<bool> getSelection(Handle<GCNNode> gradientNode, Handle<GCNMatrix> matrix) {
			return makeLambda(gradientNode, matrix, [&](Handle<GCNNode> &gradientNode, Handle<GCNMatrix> &matrix) {
				return true;
			});
		}

		Lambda<Handle<GCNNode>> getProjection(Handle<GCNNode> gradientNode, Handle<GCNMatrix> matrix) {
			return makeLambda(gradientNode, matrix,[](Handle<GCNNode> &gradientNode, Handle<GCNMatrix> &matrix) {

				auto id = gradientNode->nodeID;
				auto rows = matrix->numRows;
				auto columns = matrix->numCols;

				Handle<GCNNode> outNode = makeObject<GCNNode>(id, rows);

				float *gradientData = gradientNode->data->c_ptr();
				float *matrixData = matrix->data->c_ptr();
				float *outputData = outNode->data->c_ptr();

				// already replaced by MKL
				for (uint32_t i = 0; i < rows; ++i) {
					for (uint32_t j = 0; j < columns; ++j) {
						outputData[i] += gradientData[j] * matrixData[i * columns + j];
					}
				}

				return outNode;
			});
		}
	};
}