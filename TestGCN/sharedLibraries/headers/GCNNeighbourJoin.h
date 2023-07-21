#pragma once

#include <Object.h>
#include <PDBVector.h>
#include <LambdaCreationFunctions.h>
#include "GCNNode.h"
#include "GCNEdge.h"
#include "JoinComp.h"

namespace pdb::gcn {

        // This class implements the join between node, edges and node again*/
        class GCNNeighbourJoin : public JoinComp<GCNNeighbourJoin, GCNNode, GCNNode, GCNEdge, GCNNode> {

        public:

            ENABLE_DEEP_COPY

            GCNNeighbourJoin() = default;

            /* Join condition */
            //
            Lambda<bool> getSelection(Handle<GCNNode> sourceNode, Handle<GCNEdge> edge, Handle<GCNNode> destNode) {
                return (makeLambdaFromMember(sourceNode,nodeID) ==
                        makeLambdaFromMember(edge, source)) &&
                       (makeLambdaFromMember(edge, destination) ==
                        makeLambdaFromMember(destNode, nodeID));
            }

            Lambda<Handle<GCNNode>> getProjection(Handle<GCNNode> sourceNode, Handle<GCNEdge> edge, Handle<GCNNode> destNode) {
                return makeLambda(sourceNode, edge, destNode,[](Handle<GCNNode> &sourceNode, Handle<GCNEdge> &edge, Handle<GCNNode> &destNode) {

                  auto sourceID = sourceNode->nodeID;
                  auto featureSize = sourceNode->data->size();
                  auto destID = destNode->nodeID;
                  auto weight = edge->weight;
                  Handle<GCNNode> out = makeObject<GCNNode>(sourceID, featureSize);

                  float *in2Data = destNode->data->c_ptr();
                  float *outData = out->data->c_ptr();
                  for (int i = 0; i< featureSize;i++){
                      outData[i] = in2Data[i] * weight;
                  }
                  return out;
                });
            }
        };
    }