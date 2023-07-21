
#pragma once

#include <Object.h>
#include <PDBVector.h>
#include <LambdaCreationFunctions.h>
#include "GCNNode.h"
#include "GCNEdge.h"
#include "JoinComp.h"

namespace pdb::gcn {

	// This class implements the join between node, edges and node again*/
	class GCNZ2GradientJoin : public JoinComp<GCNZ2GradientJoin, GCNNode, GCNNode, GCNEdge, GCNNode> {

	public:

		ENABLE_DEEP_COPY

		GCNZ2GradientJoin() = default;

		/* Join condition */
		// the reverse condition of GCNNeighbourJoin.
		Lambda<bool> getSelection(Handle<GCNNode> sourceNode, Handle<GCNEdge> edge, Handle<GCNNode> destNode) {
			return (makeLambdaFromMember(sourceNode,nodeID) ==
					makeLambdaFromMember(edge, destination)) &&
				   (makeLambdaFromMember(edge, source) ==
					makeLambdaFromMember(destNode, nodeID));
		}

		Lambda<Handle<GCNNode>> getProjection(Handle<GCNNode> sourceNode, Handle<GCNEdge> edge, Handle<GCNNode> destNode) {
			return makeLambda(sourceNode, edge, destNode,[](Handle<GCNNode> &sourceNode, Handle<GCNEdge> &edge, Handle<GCNNode> &destNode) {

				auto sourceID = sourceNode->nodeID;
				auto featureSize = sourceNode->data->size();

				Handle<GCNNode> out = makeObject<GCNNode>(sourceID, featureSize);

				float *in2Data = destNode->data->c_ptr();
				float *outData = out->data->c_ptr();

				memcpy(outData, in2Data, sizeof(float) * featureSize);
				return out;
			});
		}
	};
}