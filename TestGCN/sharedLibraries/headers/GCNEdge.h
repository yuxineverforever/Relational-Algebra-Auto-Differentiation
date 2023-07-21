#pragma once

#include <Object.h>
#include <PDBVector.h>

namespace pdb::gcn {

        class GCNEdge : public pdb::Object {

        public:

            ENABLE_DEEP_COPY

            GCNEdge() = default;

            GCNEdge(int sID, int dID, float w = 1.0): source(sID), destination(dID), weight(w){
            }

            int source;
			int destination;
            float weight;
        };

    }