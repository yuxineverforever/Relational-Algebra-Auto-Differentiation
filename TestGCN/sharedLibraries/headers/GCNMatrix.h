#pragma once


#include <Object.h>
#include <PDBVector.h>

namespace pdb::gcn {

    class GCNMatrix: public pdb::Object{
    public:

        ENABLE_DEEP_COPY

        GCNMatrix() = default;
        GCNMatrix(uint32_t featureNum1,uint32_t featureNum2):numRows(featureNum1),numCols(featureNum2){
            data = makeObject<Vector<float>>(numRows * numCols, numRows * numCols);
            dummy_key = (int)numRows * (int)numCols;
        }

        /**
         * The num of rows
         */
        uint32_t numRows{};

        /**
         * The num of cols
         */
        uint32_t numCols{};

        /**
         * it is a dummy key
         */
        int dummy_key{};

        /**
         * The values of the matrix
        */
        Handle<Vector<float>> data;

        int& getKey() {
            return dummy_key;
        }

        Handle<Vector<float>>& getValue(){
            return data;
        }

        Vector<float>& getValueRef(){
            return *data;
        }

    };
}