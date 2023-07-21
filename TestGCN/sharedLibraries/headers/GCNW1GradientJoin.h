#pragma once

#include <Object.h>
#include <PDBVector.h>
#include <LambdaCreationFunctions.h>
#include "GCNNode.h"
#include "GCNEdge.h"
#include "JoinComp.h"

namespace pdb::gcn {

	// This class implements the gradients for W2.
	class GCNW1GradientJoin : public JoinComp<GCNW1GradientJoin, GCNNode, GCNNode, GCNNode> {

	public:

		ENABLE_DEEP_COPY

		GCNW1GradientJoin() = default;

		/* Join condition */
		Lambda<bool> getSelection(Handle<GCNNode> inputNode, Handle<GCNNode> gradientNode) {
			return (makeLambdaFromMember(inputNode,nodeID) ==
					makeLambdaFromMember(gradientNode, nodeID));
		}

		Lambda<Handle<GCNNode>> getProjection(Handle<GCNNode> inputNode, Handle<GCNNode> gradientNode) {
			return makeLambda(inputNode, gradientNode,[](Handle<GCNNode> &inputNode, Handle<GCNNode> &gradientNode) {

				auto id = inputNode->nodeID;
				auto rows = inputNode->data->size();
				auto columns = gradientNode->data->size();

				Handle<GCNNode> outNode = makeObject<GCNNode>(id, rows*columns);

				float *inputData = inputNode->data->c_ptr();
				float *gradientData = gradientNode->data->c_ptr();
				float *outputData = outNode->data->c_ptr();

				//TODO replace this with mkl
				// a tall vector multiply with a wide vector. The result is a matrix.
				for (uint32_t i = 0; i < rows; ++i) {
					for (uint32_t j = 0; j < columns; ++j) {
						outputData[i * columns + j] = inputData[i] * gradientData[j];
					}
				}

				return outNode;
			});
		}
	};
}