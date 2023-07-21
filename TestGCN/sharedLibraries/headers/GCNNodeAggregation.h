
#pragma once

#include "LambdaCreationFunctions.h"
#include "GCNNode.h"
#include "AggregateComp.h"
#include <DeepCopy.h>
#include "IntIntVectorPair.h"

namespace pdb::gcn {


        // template<typename Derived, class OutputClass, class InputClass, class KeyClass, class ValueClass>
        class GCNNodeAggregation : public AggregateComp<GCNNodeAggregation, GCNNode, GCNNode, int, Vector<float>> {
        public:

            ENABLE_DEEP_COPY

            GCNNodeAggregation() = default;

            // the key type must have == and size_t hash () defined
            static Lambda<int> getKeyProjection(Handle<GCNNode> aggMe) {
                return makeLambda(aggMe, [](Handle<GCNNode>& aggMe){ return aggMe->nodeID;});
            }

            // the value type must have + defined
            // Handle<Vector<float>>
            static Lambda<Vector<float>> getValueProjection(Handle<GCNNode> aggMe) {
                return makeLambdaFromMethod(aggMe, getValueRef);
            }
   };
}



