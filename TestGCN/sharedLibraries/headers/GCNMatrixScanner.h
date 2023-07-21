

#ifndef PDB_GCNMATRIXSCANNER_H
#define PDB_GCNMATRIXSCANNER_H

#include "GCNMatrix.h"
#include "SetScanner.h"

namespace pdb::gcn {

        class GCNMatrixScanner: public pdb::SetScanner<pdb::gcn::GCNMatrix> {
        public:
            /**
             * The default constructor
             */
            GCNMatrixScanner() = default;

            GCNMatrixScanner(const std::string &db, const std::string &set) : SetScanner(db, set) {}

            ENABLE_DEEP_COPY
        };
    }


#endif //PDB_GCNMATRIXSCANNER_H
