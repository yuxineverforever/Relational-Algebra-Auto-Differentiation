#pragma once

#include <Object.h>
#include <PDBVector.h>
#include <LambdaCreationFunctions.h>
#include "GCNNode.h"
#include "GCNEdge.h"
#include "JoinComp.h"
#include "math.h"

namespace pdb::gcn {

    // This class implements the gradients for Z4.
    // or Gradient for Softmax / MSE.
    class GCNNodeLossJoin : public JoinComp<GCNNodeLossJoin, GCNNode, GCNNode, GCNNode> {

    public:

        ENABLE_DEEP_COPY

        GCNNodeLossJoin() = default;

        /* Join condition */
        Lambda<bool> getSelection(Handle<GCNNode> outputNode, Handle<GCNNode> labelNode) {
            return (makeLambdaFromMember(outputNode, nodeID) ==
                    makeLambdaFromMember(labelNode, nodeID));
        }

        Lambda<Handle<GCNNode>> getProjection(Handle<GCNNode> outputNode, Handle<GCNNode> labelNode) {
            return makeLambda(outputNode,labelNode,[](Handle<GCNNode> &outputNode, Handle<GCNNode> &labelNode) {

                auto id = outputNode->nodeID;
                auto length = labelNode->data->size();

                Handle<GCNNode> outNode = makeObject<GCNNode>(id, 2);

                float* labelData = labelNode->data->c_ptr();
                float* outputData = outputNode->data->c_ptr();
                float* lossData = outNode->data->c_ptr();

                int predicatedClass = 0;
                int labelClass = 0;
                float predicatedProb = std::numeric_limits<float>::min();
                float labelProb = std::numeric_limits<float>::min();
                for (int i = 0; i < length; i++){
                    lossData[0] +=  log(outputData[i]) * labelData[i];
                    if (outputData[i] >= predicatedProb){
                        predicatedProb = outputData[i];
                        predicatedClass = i;
                    }
                    if (labelData[i] >= labelProb){
                        labelProb = labelData[i];
                        labelClass = i;
                    }
                }

                // already replaced by MKL
                lossData[0] = (-1) * lossData[0] / 2708;

                if (predicatedClass == labelClass){
                    lossData[1] = 1.0;
                } else {
                    lossData[1] = 0.0;
                }
                //std::cout << "loss data: " << lossData[0] << std::endl;
                return outNode;
            });
        }
    };
}