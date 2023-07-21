
#pragma once

#include <Object.h>
#include <PDBVector.h>
#include <LambdaCreationFunctions.h>
#include "GCNNode.h"
#include "GCNEdge.h"
#include "GCNMatrix.h"
#include "JoinComp.h"

namespace pdb::gcn {

	// This class implements the join between node, edges and node again*/
	class GCNZ1GradientJoin : public JoinComp<GCNZ1GradientJoin, GCNNode, GCNNode, GCNNode> {

	public:

		ENABLE_DEEP_COPY

		GCNZ1GradientJoin() = default;

		/* Join condition */
		// the reverse condition of GCNNeighbourJoin.
		Lambda<bool> getSelection(Handle<GCNNode> gradientNode, Handle<GCNNode> z1) {
				return (makeLambdaFromMember(gradientNode, nodeID) ==
						makeLambdaFromMember(z1, nodeID));
		};

		Lambda<Handle<GCNNode>> getProjection(Handle<GCNNode> gradientNode, Handle<GCNNode> z1) {
			return makeLambda(gradientNode, z1,[](Handle<GCNNode> &gradientNode, Handle<GCNNode> &z1) {

				auto sourceID = gradientNode->nodeID;
				auto vectorSize = gradientNode->data->size();

				Handle<GCNNode> out = makeObject<GCNNode>(sourceID, vectorSize);

				float * gradientData = gradientNode->data->c_ptr();
				float * z1Data = z1->data->c_ptr();
				float * outData = out->data->c_ptr();

				for (auto i =0; i<vectorSize;i++){
					outData[i] = z1Data[i] * gradientData[i];
				}
				return out;
			});
		}
	};
}