#pragma once

#include <Object.h>
#include <PDBVector.h>

namespace pdb{

    inline Vector<float>& operator+(Vector<float>& lhs, Vector<float>& rhs) {
        int size = lhs.size();
        if (size != rhs.size()) {
            std::cout << "You cannot add two vectors in different sizes!" << std::endl;
            return lhs;
        }
        for (int i = 0; i < size; ++i) {
            lhs[i] = lhs[i] + rhs[i];
        }
        return lhs;
    }
}

namespace pdb::gcn {

        class GCNNode : public pdb::Object {
        public:
            /**
             * The default constructor
             */
            GCNNode() = default;

            GCNNode(int nodeID, uint32_t featureSize) : nodeID(nodeID), featureSize(featureSize){
                // allocate the data
                data = makeObject<Vector<float>>(featureSize, featureSize);
            }

            ENABLE_DEEP_COPY

            /**
             * The id of the node
             */
            int nodeID{};

            /**
             * the length of the embedding vector of this node
             *
             * another way to get @featureSize is through data.size()
             */
            uint32_t  featureSize{};

            /**
             * embedding vector itself.
             */
            Handle<Vector<float>> data;


            int& getKey() {
                return nodeID;
            }

            Handle<Vector<float>>& getValue(){
                return data;
            }

            Vector<float>& getValueRef(){
                return *data;
            }

        };

}