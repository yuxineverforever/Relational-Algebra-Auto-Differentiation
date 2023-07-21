#pragma once

#include "GCNNode.h"
#include "SetScanner.h"

namespace pdb {

// the sub namespace
    namespace gcn {

        /**
        * The GCNNode scanner
        */
        class GCNNodeScanner : public pdb::SetScanner<pdb::gcn::GCNNode> {
        public:
            /**
             * The default constructor
             */
            GCNNodeScanner() = default;

            GCNNodeScanner(const std::string &db, const std::string &set) : SetScanner(db, set) {}

            ENABLE_DEEP_COPY

        };

    }

}
