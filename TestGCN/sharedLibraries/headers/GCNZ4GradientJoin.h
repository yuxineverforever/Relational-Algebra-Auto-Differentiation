#pragma once

#include <Object.h>
#include <PDBVector.h>
#include <LambdaCreationFunctions.h>
#include "GCNNode.h"
#include "GCNEdge.h"
#include "JoinComp.h"

namespace pdb::gcn {

	// This class implements the gradients for Z4.
	// or Gradient for Softmax / MSE.
	class GCNZ4GradientJoin : public JoinComp<GCNZ4GradientJoin, GCNNode, GCNNode, GCNNode> {

	public:

		ENABLE_DEEP_COPY

		GCNZ4GradientJoin() = default;

		/* Join condition */
		Lambda<bool> getSelection(Handle<GCNNode> featureNode, Handle<GCNNode> labelNode) {
			return (makeLambdaFromMember(featureNode,nodeID) ==
					makeLambdaFromMember(labelNode, nodeID));
		}

		Lambda<Handle<GCNNode>> getProjection(Handle<GCNNode> featureNode, Handle<GCNNode> labelNode) {
			return makeLambda(featureNode,labelNode,[](Handle<GCNNode> &featureNode, Handle<GCNNode> &labelNode) {

				auto id = featureNode->nodeID;
				auto length = featureNode->data->size();

				Handle<GCNNode> outNode = makeObject<GCNNode>(id, length);

				float *labelData = labelNode->data->c_ptr();
				float *featureData = featureNode->data->c_ptr();
				float *gradientData = outNode->data->c_ptr();

				for (int i = 0; i < length; i++){
					gradientData[i] = featureData[i] - labelData[i];
				}

				return outNode;
			});
		}
	};
}