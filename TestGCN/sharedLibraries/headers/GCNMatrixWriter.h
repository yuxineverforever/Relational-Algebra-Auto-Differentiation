

#ifndef PDB_GCNMATRIXWRITER_H
#define PDB_GCNMATRIXWRITER_H


#include "SetWriter.h"
#include "GCNNode.h"
#include "GCNMatrix.h"

namespace pdb::gcn{

    class GCNMatrixWriter: public SetWriter<pdb::gcn::GCNMatrix> {
    public:
        /**
         * The default constructor
         */
        GCNMatrixWriter() = default;

        GCNMatrixWriter(const String &dbName, const String &setName) : SetWriter(dbName, setName) {}

        ENABLE_DEEP_COPY
    };
}

#endif //PDB_GCNMATRIXWRITER_H
