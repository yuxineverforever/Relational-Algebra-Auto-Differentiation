

#ifndef PDB_GCNNODEWRITER_H
#define PDB_GCNNODEWRITER_H

#include "SetWriter.h"
#include "GCNNode.h"

namespace pdb::gcn{

    class GCNNodeWriter: public SetWriter<pdb::gcn::GCNNode> {
    public:
        /**
         * The default constructor
         */
        GCNNodeWriter() = default;

        GCNNodeWriter(const String &dbName, const String &setName) : SetWriter(dbName, setName) {}

        ENABLE_DEEP_COPY
    };
}


#endif //PDB_GCNNODEWRITER_H
