
#pragma once

#include "GCNEdge.h"
#include "SetScanner.h"

namespace pdb::gcn {
        /**
        * The GCNEdge scanner
        */
        class GCNEdgeScanner: public pdb::SetScanner<pdb::gcn::GCNEdge> {
        public:
            /**
             * The default constructor
             */
            GCNEdgeScanner() = default;

            GCNEdgeScanner(const std::string &db, const std::string &set) : SetScanner(db, set) {}

            ENABLE_DEEP_COPY
        };
}