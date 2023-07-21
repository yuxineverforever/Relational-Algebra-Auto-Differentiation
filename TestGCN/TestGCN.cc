#include <fstream>
#include <vector>
#include <PDBClient.h>
#include <random>
#include "sharedLibraries/headers/GCNMatrix.h"
#include "sharedLibraries/headers/GCNNode.h"
#include "sharedLibraries/headers/GCNEdge.h"
#include "sharedLibraries/headers/GCNNeighbourJoin.h"
#include "sharedLibraries/headers/GCNTransformReLUJoin.h"
#include "sharedLibraries/headers/GCNTransformSoftmaxJoin.h"
#include "sharedLibraries/headers/GCNNodeScanner.h"
#include "sharedLibraries/headers/GCNEdgeScanner.h"
#include "sharedLibraries/headers/GCNNodeAggregation.h"
#include "sharedLibraries/headers/GCNNodeWriter.h"
#include "sharedLibraries/headers/GCNMatrixScanner.h"
#include "sharedLibraries/headers/GCNZ4GradientJoin.h"
#include "sharedLibraries/headers/GCNW2GradientJoin.h"
#include "sharedLibraries/headers/GCNW2GradientAggregation.h"
#include "sharedLibraries/headers/GCNZ3GradientJoin.h"
#include "sharedLibraries/headers/GCNZ2GradientJoin.h"
#include "sharedLibraries/headers/GCNZ1GradientJoin.h"
#include "sharedLibraries/headers/GCNW1GradientJoin.h"
#include "sharedLibraries/headers/GCNW1GradientAggregation.h"
#include "sharedLibraries/headers/GCNTransformReLUDerivateJoin.h"
#include "sharedLibraries/headers/GCNMatrixUpdateJoin.h"
#include "sharedLibraries/headers/GCNMatrixWriter.h"
#include "sharedLibraries/headers/GCNNodeLossJoin.h"
#include "sharedLibraries/headers/GCNNodeLossAggregation.h"

using namespace pdb;
using namespace gcn;
using entityID = int32_t;
using labelID = std::string;

// total number of features.
const int feature_count = 1433;
const int layer1_embedding_count = 16;
// total number of labels.
const int labels_count = 7;


int iterations = 100;
const size_t blockSize = 64;
bool doNotPrint = false;


std::map<entityID, std::set<entityID>> edges;
std::vector<std::pair<entityID,std::vector<float>>> features;
std::vector<std::pair<entityID,std::vector<int32_t>>> labels;

void normalize(std::vector<float> &vec){
    float m = 0;
    for (auto& v:vec){
        m += v*v;
    }
    m = std::sqrt(m);
    for (auto& v:vec){
        v = v/m;
    }
}

void initEdge(PDBClient& client, const std::string& setName) {

	// make the allocation block
	const pdb::UseTemporaryAllocationBlock tempBlock{blockSize * 1024 * 1024};

	// put the chunks here
	Handle<Vector<Handle<GCNEdge>>> data = pdb::makeObject<Vector<Handle<GCNEdge>>>();

	for (auto& edge: edges){
        auto source = edge.first;
        float weight = ((float)1.0)/((float)(edge.second.size()*edge.second.size()));
        for (auto & dest : edge.second){
            Handle<GCNEdge> myEdge = makeObject<GCNEdge>(source, dest, weight);
            data->push_back(myEdge);
        }
	}

	// init the records
	getRecord(data);

	// send the data a bunch of times
	client.sendData<GCNEdge>("gcn", setName, data);
}

void initFeature(PDBClient& client, const std::string& setName) {

	// make the allocation block
	const pdb::UseTemporaryAllocationBlock tempBlock{blockSize * 1024 * 1024};

	// put the chunks here
	Handle<Vector<Handle<GCNNode>>> data = pdb::makeObject<Vector<Handle<GCNNode>>>();

	for (const auto& node_feat : features){
		auto id = node_feat.first;
		auto& feat = node_feat.second;
		Handle<GCNNode> myNode = makeObject<GCNNode>(id,feature_count);
		float* values = myNode->data->c_ptr();
		for (int v = 0; v < feature_count; v++){
			values[v] = 1.0f * feat[v];
		}
		data->push_back(myNode);
	}
	getRecord(data);
	client.sendData<GCNNode>("gcn", setName, data);
}

void initLabel(PDBClient& client, const std::string& setName){

	// make the allocation block
	const pdb::UseTemporaryAllocationBlock tempBlock{blockSize * 1024 * 1024};

	// put the chunks here
	Handle<Vector<Handle<GCNNode>>> data = pdb::makeObject<Vector<Handle<GCNNode>>>();

	for (const auto& node_label : labels){
		auto id = node_label.first;
		auto& label = node_label.second;
		Handle<GCNNode> myNode = makeObject<GCNNode>(id,labels_count);
		float* values = myNode->data->c_ptr();
		for (int v = 0; v < labels_count; v++){
			values[v] = 1.0f * label[v];
		}
		data->push_back(myNode);
	}

	getRecord(data);
	client.sendData<GCNNode>("gcn", setName, data);
}

void initMatrix(PDBClient& client, const std::string& setName, int32_t rows, int32_t columns){

    const float range_from  = -1.0*sqrt(1.0/columns);
    const float range_to    = 1.0*sqrt(1.0/columns);
    std::random_device                  rand_dev;
    std::mt19937                        generator(rand_dev());
    std::uniform_real_distribution<float>  distr(range_from, range_to);

	// make the allocation block
	const pdb::UseTemporaryAllocationBlock tempBlock{blockSize * 1024 * 1024};

	// put the chunks here
	Handle<Vector<Handle<GCNMatrix>>> data = pdb::makeObject<Vector<Handle<GCNMatrix>>>();

	Handle<GCNMatrix> myMatrix = makeObject<GCNMatrix>(rows,columns);

	float *vals = myMatrix->data->c_ptr();
	for(int i = 0; i < rows*columns; ++i) {
		vals[i] = distr(generator);
	}
	data->push_back(myMatrix);
	getRecord(data);
	client.sendData<GCNMatrix>("gcn", setName, data);
}


std::vector<int32_t> encode_onehot(std::string label){
	std::vector<int32_t> encoding;
	for (int i =0;i<labels_count;i++){
		if (all_labels[i] == label){
			encoding.push_back(1);
		} else {
			encoding.push_back(0);
		}
	}
	return encoding;
}

int read_data(){

	std::ifstream cites("./applications/TestGCN/edges");
	entityID source;
	entityID dest;
	while(!cites.eof()){
		cites>>source;
		cites>>dest;
        if (edges.find(source)!= edges.end()){
            edges[source].insert(dest);
        } else {
            std::set<entityID> newSet;
            newSet.insert(dest);
            edges[source] = newSet;
        }
	}
	cites.close();

    // add self loop. add self-connections for the edges.
    for (auto & edgeSet : edges){
        if (edgeSet.second.find(edgeSet.first) == edgeSet.second.end()){
            edgeSet.second.insert(edgeSet.first);
        }
    }

	std::ifstream content("./applications/TestGCN/features");

	int32_t binary_feature;
	entityID id;
	std::string label;
	while (!content.eof()) {
		std::vector<float> feature;
		content>>id;
		for (int i = 0; i<feature_count; i++){
			content>>binary_feature;
			feature.push_back((float)binary_feature);
		}
		content >> label;
        normalize(feature);
		features.emplace_back(id,feature);
		labels.emplace_back(id,std::move(encode_onehot(label)));
	}
	content.close();
	std::cout << "read data done.\n";
	return 0;
}

int main(int argc, char *argv[]) {
    read_data();
    // make a client
    pdb::PDBClient pdbClient(8108, "localhost");

    // now, register a type for user data
    pdbClient.registerType("libraries/libGCNNode.so");
    pdbClient.registerType("libraries/libGCNEdge.so");
    pdbClient.registerType("libraries/libGCNMatrix.so");
    pdbClient.registerType("libraries/libGCNMatrixUpdateJoin.so");
    pdbClient.registerType("libraries/libGCNMatrixWriter.so");
    pdbClient.registerType("libraries/libGCNNeighbourJoin.so");
    pdbClient.registerType("libraries/libGCNTransformReLUJoin.so");
    pdbClient.registerType("libraries/libGCNTransformSoftmaxJoin.so");
    pdbClient.registerType("libraries/libGCNTransformReLUDerivateJoin.so");
    pdbClient.registerType("libraries/libGCNNodeScanner.so");
    pdbClient.registerType("libraries/libGCNEdgeScanner.so");
    pdbClient.registerType("libraries/libGCNMatrixScanner.so");
    pdbClient.registerType("libraries/libGCNNodeAggregation.so");
    pdbClient.registerType("libraries/libGCNNodeWriter.so");
    pdbClient.registerType("libraries/libGCNNodeLossJoin.so");
    pdbClient.registerType("libraries/libGCNNodeLossAggregation.so");
    pdbClient.registerType("libraries/libGCNW1GradientJoin.so");
    pdbClient.registerType("libraries/libGCNW1GradientAggregation.so");
    pdbClient.registerType("libraries/libGCNW2GradientJoin.so");
    pdbClient.registerType("libraries/libGCNW2GradientAggregation.so");
    pdbClient.registerType("libraries/libGCNZ2GradientJoin.so");
    pdbClient.registerType("libraries/libGCNZ1GradientJoin.so");
    pdbClient.registerType("libraries/libGCNZ3GradientJoin.so");
    pdbClient.registerType("libraries/libGCNZ4GradientJoin.so");

    // now, create a new database
    pdbClient.createDatabase("gcn");

    // now, create the input and output sets
    pdbClient.createSet<gcn::GCNNode>("gcn", "input_node");
    pdbClient.createSet<gcn::GCNEdge>("gcn", "input_edge");
    pdbClient.createSet<gcn::GCNNode>("gcn","labels");

    pdbClient.createSet<gcn::GCNMatrix>("gcn", "w1");
    pdbClient.createSet<gcn::GCNMatrix>("gcn", "w2");
    pdbClient.createSet<gcn::GCNMatrix>("gcn","w1_updated");
    pdbClient.createSet<gcn::GCNMatrix>("gcn","w2_updated");

    pdbClient.createSet<gcn::GCNNode>("gcn","layer1_output");
    pdbClient.createSet<gcn::GCNNode>("gcn", "layer2_output");
    pdbClient.createSet<gcn::GCNNode>("gcn", "loss");


    pdbClient.createSet<gcn::GCNNode>("gcn", "Z3");
    pdbClient.createSet<gcn::GCNNode>("gcn", "Z1");

    pdbClient.createSet<gcn::GCNNode>("gcn", "Z1_gradient");
    pdbClient.createSet<gcn::GCNNode>("gcn", "Z2_gradient");
    pdbClient.createSet<gcn::GCNNode>("gcn", "Z3_gradient");
    pdbClient.createSet<gcn::GCNNode>("gcn", "Z4_gradient");


    initEdge(pdbClient, "input_edge");
    initFeature(pdbClient, "input_node");
    initLabel(pdbClient, "labels");

    initMatrix(pdbClient, "w1", feature_count, layer1_embedding_count);
    initMatrix(pdbClient, "w2", layer1_embedding_count, labels_count);

    for (int iter = 0 ; iter < iterations; iter++){

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        {
            const UseTemporaryAllocationBlock tempBlock{1024 * 1024 * 128};

            Handle<Computation> n1 = makeObject<gcn::GCNNodeScanner>("gcn", "input_node");
            Handle<Computation> n2 = makeObject<gcn::GCNNodeScanner>("gcn", "input_node");
            Handle<Computation> e = makeObject<gcn::GCNEdgeScanner>("gcn", "input_edge");


            Handle<Computation> neighbourjoin = makeObject<gcn::GCNNeighbourJoin>();
            neighbourjoin->setInput(0, n1);
            neighbourjoin->setInput(1, e);
            neighbourjoin->setInput(2, n2);


            Handle<Computation> neighbourAggregation = makeObject<gcn::GCNNodeAggregation>();
            neighbourAggregation->setInput(neighbourjoin);

            Handle<Computation> myWriter = makeObject<gcn::GCNNodeWriter>("gcn", "Z1");
            myWriter->setInput(neighbourAggregation);

            bool success = pdbClient.executeComputations({myWriter});
        }
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::cout << "Time for 1 stage: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() << std::endl;
        begin = std::chrono::steady_clock::now();
        {
            const UseTemporaryAllocationBlock tempBlock{1024 * 1024 * 128};

            Handle<Computation> n1 = makeObject<gcn::GCNNodeScanner>("gcn", "Z1");
            Handle<Computation> m1 = makeObject<gcn::GCNMatrixScanner>("gcn", "w1");

            Handle<Computation> transformjoin = makeObject<gcn::GCNTransformReLUJoin>();
            transformjoin->setInput(0, n1);
            transformjoin->setInput(1, m1);

            Handle<Computation> myWriter = makeObject<gcn::GCNNodeWriter>("gcn", "layer1_output");
            myWriter->setInput(transformjoin);
            bool success = pdbClient.executeComputations({myWriter});
        }
        end = std::chrono::steady_clock::now();
        std::cout << "Time for 2 stage: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() << std::endl;

        begin = std::chrono::steady_clock::now();
        {
            const UseTemporaryAllocationBlock tempBlock{1024 * 1024 * 128};


            Handle<Computation> n1 = makeObject<gcn::GCNNodeScanner>("gcn", "layer1_output");
            Handle<Computation> n2 = makeObject<gcn::GCNNodeScanner>("gcn", "layer1_output");
            Handle<Computation> e = makeObject<gcn::GCNEdgeScanner>("gcn", "input_edge");

            Handle<Computation> neighbourjoin = makeObject<gcn::GCNNeighbourJoin>();
            neighbourjoin->setInput(0, n1);
            neighbourjoin->setInput(1, e);
            neighbourjoin->setInput(2, n2);

            Handle<Computation> neighbourAggregation = makeObject<gcn::GCNNodeAggregation>();
            neighbourAggregation->setInput(neighbourjoin);

            Handle<Computation> myWriter = makeObject<gcn::GCNNodeWriter>("gcn", "Z3");
            myWriter->setInput(neighbourAggregation);
            bool success = pdbClient.executeComputations({myWriter});
        }
        end = std::chrono::steady_clock::now();
        std::cout << "Time for 3 stage: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() << std::endl;

        begin = std::chrono::steady_clock::now();
        {
            const UseTemporaryAllocationBlock tempBlock{1024 * 1024 * 128};
            Handle<Computation> n3 = makeObject<gcn::GCNNodeScanner>("gcn", "Z3");
            Handle<Computation> m2 = makeObject<gcn::GCNMatrixScanner>("gcn", "w2");

            Handle<Computation> transformjoin = makeObject<gcn::GCNTransformSoftmaxJoin>();
            transformjoin->setInput(0, n3);
            transformjoin->setInput(1, m2);

            Handle<Computation> myWriter = makeObject<gcn::GCNNodeWriter>("gcn", "layer2_output");
            myWriter->setInput(transformjoin);

            bool success = pdbClient.executeComputations({myWriter});
        }
        end = std::chrono::steady_clock::now();
        std::cout << "Time for 4 stage: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() << std::endl;


        begin = std::chrono::steady_clock::now();
        {
            const UseTemporaryAllocationBlock tempBlock{1024 * 1024 * 128};
            Handle<Computation> n = makeObject<gcn::GCNNodeScanner>("gcn", "layer2_output");
            Handle<Computation> l = makeObject<gcn::GCNNodeScanner>("gcn", "labels");


            Handle<Computation> lossjoin = makeObject<gcn::GCNNodeLossJoin>();
            lossjoin->setInput(0, n);
            lossjoin->setInput(1, l);

            Handle<Computation> lossagg = makeObject<gcn::GCNNodeLossAggregation>();
            lossagg->setInput(lossjoin);


            Handle<Computation> myWriter = makeObject<gcn::GCNNodeWriter>("gcn", "loss");
            myWriter->setInput(lossagg);

            bool success = pdbClient.executeComputations({myWriter});
        }
        end = std::chrono::steady_clock::now();
        std::cout << "Time for 5 stage: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() << std::endl;
        auto it = pdbClient.getSetIterator<GCNMatrix>("gcn", "loss");
        int32_t count = 0;
        while(it->hasNextRecord()) {
            auto r = it->getNextRecord();
            count++;


            if(doNotPrint) {
                continue;
            }

            float* values = r->data->c_ptr();
            std::cout << "Loss: " << values[0] << "\n\n\n\n";
            std::cout << "Accuracy: " << values[1]/2708 << "\n\n\n\n";
            std::cout  << "\n\n";
        }
        begin = std::chrono::steady_clock::now();
        {
            const UseTemporaryAllocationBlock tempBlock{1024 * 1024 * 128};


            Handle<Computation> n = makeObject<gcn::GCNNodeScanner>("gcn", "layer2_output");
            Handle<Computation> l = makeObject<gcn::GCNNodeScanner>("gcn", "labels");


            Handle<Computation> Z4GradientJoin = makeObject<gcn::GCNZ4GradientJoin>();
            Z4GradientJoin->setInput(0, n);
            Z4GradientJoin->setInput(1, l);

            Handle<Computation> myWriter = makeObject<gcn::GCNNodeWriter>("gcn", "Z4_gradient");
            myWriter->setInput(Z4GradientJoin);

            bool success = pdbClient.executeComputations({myWriter});
        }
        end = std::chrono::steady_clock::now();
        std::cout << "Time for 6 stage: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() << std::endl;

        begin = std::chrono::steady_clock::now();
        {
            const UseTemporaryAllocationBlock tempBlock{1024 * 1024 * 128};

            Handle<Computation> z3 = makeObject<gcn::GCNNodeScanner>("gcn", "Z3");
            Handle<Computation> z4_gradient = makeObject<gcn::GCNNodeScanner>("gcn", "Z4_gradient");
            Handle<Computation> m2 = makeObject<gcn::GCNMatrixScanner>("gcn", "w2");


            Handle<Computation> W2GradientJoin = makeObject<gcn::GCNW2GradientJoin>();
            W2GradientJoin->setInput(0, z3);
            W2GradientJoin->setInput(1, z4_gradient);
            Handle<Computation> W2GradientAggregation = makeObject<gcn::GCNW2GradientAggregation>();
            W2GradientAggregation->setInput(W2GradientJoin);

            Handle<Computation> W2Updated = makeObject<gcn::GCNMatrixUpdateJoin>();
            W2Updated->setInput(0,W2GradientAggregation);
            W2Updated->setInput(1,m2);

            Handle<Computation> myWriter = makeObject<gcn::GCNMatrixWriter>("gcn", "w2_updated");
            myWriter->setInput(W2Updated);

            bool success = pdbClient.executeComputations({myWriter});
        }
        end = std::chrono::steady_clock::now();
        std::cout << "Time for 7 stage: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() << std::endl;
        begin = std::chrono::steady_clock::now();
        {
            const UseTemporaryAllocationBlock tempBlock{1024 * 1024 * 128};

            Handle<Computation> z4_gradient = makeObject<gcn::GCNNodeScanner>("gcn", "Z4_gradient");
            Handle<Computation> m2 = makeObject<gcn::GCNMatrixScanner>("gcn", "w2");

            Handle<Computation> Z3GradientJoin = makeObject<gcn::GCNZ3GradientJoin>();
            Z3GradientJoin->setInput(0, z4_gradient);
            Z3GradientJoin->setInput(1, m2);

            Handle<Computation> myWriter = makeObject<gcn::GCNNodeWriter>("gcn", "Z3_gradient");
            myWriter->setInput(Z3GradientJoin);

            bool success = pdbClient.executeComputations({myWriter});
        }
        end = std::chrono::steady_clock::now();

        std::cout << "Time for 8 stage: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() << std::endl;

        begin = std::chrono::steady_clock::now();
        {
            const UseTemporaryAllocationBlock tempBlock{1024 * 1024 * 128};


            Handle<Computation> z3_gradient = makeObject<gcn::GCNNodeScanner>("gcn", "Z3_gradient");
            Handle<Computation> e = makeObject<gcn::GCNEdgeScanner>("gcn", "input_edge");


            Handle<Computation> Z2GradientJoin = makeObject<gcn::GCNZ2GradientJoin>();
            Z2GradientJoin->setInput(0,z3_gradient);
            Z2GradientJoin->setInput(1,e);
            Z2GradientJoin->setInput(2,z3_gradient);

            Handle<Computation> Z2Aggregation = makeObject<gcn::GCNNodeAggregation>();
            Z2Aggregation->setInput(Z2GradientJoin);

            Handle<Computation> myWriter = makeObject<gcn::GCNNodeWriter>("gcn", "Z2_gradient");
            myWriter->setInput(Z2Aggregation);

            bool success = pdbClient.executeComputations({myWriter});
        }
        end = std::chrono::steady_clock::now();
        std::cout << "Time for 9 stage: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() << std::endl;

        begin = std::chrono::steady_clock::now();
        {
            const UseTemporaryAllocationBlock tempBlock{1024 * 1024 * 128};

            Handle<Computation> z2_gradient = makeObject<gcn::GCNNodeScanner>("gcn", "Z2_gradient");
            Handle<Computation> z1 = makeObject<gcn::GCNNodeScanner>("gcn", "Z1");
            Handle<Computation> m1 = makeObject<gcn::GCNMatrixScanner>("gcn", "w1");

            Handle<Computation> transformjoin = makeObject<gcn::GCNTransformReLUDerivateJoin>();
            transformjoin->setInput(0,z1);
            transformjoin->setInput(1,m1);

            Handle<Computation> Z1GradientJoin = makeObject<gcn::GCNZ1GradientJoin>();
            Z1GradientJoin->setInput(0,z2_gradient);
            Z1GradientJoin->setInput(1,transformjoin);

            Handle<Computation> myWriter = makeObject<gcn::GCNNodeWriter>("gcn", "Z1_gradient");
            myWriter->setInput(Z1GradientJoin);

            bool success = pdbClient.executeComputations({myWriter});
        }
        end = std::chrono::steady_clock::now();

        std::cout << "Time for 10 stage: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() << std::endl;
        begin = std::chrono::steady_clock::now();
        {
            const UseTemporaryAllocationBlock tempBlock{1024 * 1024 * 128};
            Handle<Computation> z1 = makeObject<gcn::GCNNodeScanner>("gcn", "Z1");
            Handle<Computation> z1_gradient = makeObject<gcn::GCNNodeScanner>("gcn", "Z1_gradient");
            Handle<Computation> m1 = makeObject<gcn::GCNMatrixScanner>("gcn", "w1");

            Handle<Computation> W1GradientJoin = makeObject<gcn::GCNW1GradientJoin>();
            W1GradientJoin->setInput(0, z1);
            W1GradientJoin->setInput(1, z1_gradient);
            Handle<Computation> W1GradientAggregation = makeObject<gcn::GCNW1GradientAggregation>();
            W1GradientAggregation->setInput(W1GradientJoin);

            Handle<Computation> W1Updated = makeObject<gcn::GCNMatrixUpdateJoin>();
            W1Updated->setInput(0, W1GradientAggregation);
            W1Updated->setInput(1, m1);

            Handle<Computation> myWriter = makeObject<gcn::GCNMatrixWriter>("gcn", "w1_updated");
            myWriter->setInput(W1Updated);
            bool success = pdbClient.executeComputations({myWriter});
        }
        end = std::chrono::steady_clock::now();

        std::cout << "Time for 11 stage: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() << std::endl;

        pdbClient.clearSet("gcn", "w1");
        pdbClient.clearSet("gcn", "w2");
        pdbClient.clearSet("gcn","layer1_output");
        pdbClient.clearSet("gcn","layer2_output");
        pdbClient.clearSet("gcn", "Z3");
        pdbClient.clearSet("gcn", "Z1");
        pdbClient.clearSet("gcn","loss");

        pdbClient.clearSet("gcn", "Z1_gradient");
        pdbClient.clearSet("gcn", "Z2_gradient");
        pdbClient.clearSet("gcn", "Z3_gradient");
        pdbClient.clearSet("gcn", "Z4_gradient");

        begin = std::chrono::steady_clock::now();
        {
            const UseTemporaryAllocationBlock tempBlock{1024 * 1024 * 128};

            Handle<Computation> n1 = makeObject<gcn::GCNNodeScanner>("gcn", "input_node");
            Handle<Computation> n2 = makeObject<gcn::GCNNodeScanner>("gcn", "input_node");
            Handle<Computation> e = makeObject<gcn::GCNEdgeScanner>("gcn", "input_edge");

            Handle<Computation> neighbourjoin = makeObject<gcn::GCNNeighbourJoin>();
            neighbourjoin->setInput(0, n1);
            neighbourjoin->setInput(1, e);
            neighbourjoin->setInput(2, n2);

            Handle<Computation> neighbourAggregation = makeObject<gcn::GCNNodeAggregation>();
            neighbourAggregation->setInput(neighbourjoin);

            Handle<Computation> myWriter = makeObject<gcn::GCNNodeWriter>("gcn", "Z1");
            myWriter->setInput(neighbourAggregation);

            bool success = pdbClient.executeComputations({myWriter});
        }
        end = std::chrono::steady_clock::now();
        std::cout << "Time for 1 stage: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() << std::endl;

        begin = std::chrono::steady_clock::now();
        {
            const UseTemporaryAllocationBlock tempBlock{1024 * 1024 * 128};

            Handle<Computation> n1 = makeObject<gcn::GCNNodeScanner>("gcn", "Z1");
            Handle<Computation> m1 = makeObject<gcn::GCNMatrixScanner>("gcn", "w1_updated");

            Handle<Computation> transformjoin = makeObject<gcn::GCNTransformReLUJoin>();
            transformjoin->setInput(0, n1);
            transformjoin->setInput(1, m1);

            Handle<Computation> myWriter = makeObject<gcn::GCNNodeWriter>("gcn", "layer1_output");
            myWriter->setInput(transformjoin);
            bool success = pdbClient.executeComputations({myWriter});
        }
        end = std::chrono::steady_clock::now();
        std::cout << "Time for 2 stage: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() << std::endl;

        begin = std::chrono::steady_clock::now();
        {
            const UseTemporaryAllocationBlock tempBlock{1024 * 1024 * 128};

            Handle<Computation> n1 = makeObject<gcn::GCNNodeScanner>("gcn", "layer1_output");
            Handle<Computation> n2 = makeObject<gcn::GCNNodeScanner>("gcn", "layer1_output");
            Handle<Computation> e = makeObject<gcn::GCNEdgeScanner>("gcn", "input_edge");

            Handle<Computation> neighbourjoin = makeObject<gcn::GCNNeighbourJoin>();
            neighbourjoin->setInput(0, n1);
            neighbourjoin->setInput(1, e);
            neighbourjoin->setInput(2, n2);

            Handle<Computation> neighbourAggregation = makeObject<gcn::GCNNodeAggregation>();
            neighbourAggregation->setInput(neighbourjoin);

            Handle<Computation> myWriter = makeObject<gcn::GCNNodeWriter>("gcn", "Z3");
            myWriter->setInput(neighbourAggregation);
            bool success = pdbClient.executeComputations({myWriter});
        }
        end = std::chrono::steady_clock::now();
        std::cout << "Time for 3 stage: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() << std::endl;

        begin = std::chrono::steady_clock::now();
        {
            const UseTemporaryAllocationBlock tempBlock{1024 * 1024 * 128};
            Handle<Computation> n3 = makeObject<gcn::GCNNodeScanner>("gcn", "Z3");
            Handle<Computation> m2 = makeObject<gcn::GCNMatrixScanner>("gcn", "w2_updated");

            Handle<Computation> transformjoin = makeObject<gcn::GCNTransformSoftmaxJoin>();
            transformjoin->setInput(0, n3);
            transformjoin->setInput(1, m2);

            Handle<Computation> myWriter = makeObject<gcn::GCNNodeWriter>("gcn", "layer2_output");
            myWriter->setInput(transformjoin);

            bool success = pdbClient.executeComputations({myWriter});
        }
        end = std::chrono::steady_clock::now();
        std::cout << "Time for 4 stage: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() << std::endl;

        begin = std::chrono::steady_clock::now();
        {
            const UseTemporaryAllocationBlock tempBlock{1024 * 1024 * 128};
            Handle<Computation> n = makeObject<gcn::GCNNodeScanner>("gcn", "layer2_output");
            Handle<Computation> l = makeObject<gcn::GCNNodeScanner>("gcn", "labels");


            Handle<Computation> lossjoin = makeObject<gcn::GCNNodeLossJoin>();
            lossjoin->setInput(0, n);
            lossjoin->setInput(1, l);

            Handle<Computation> lossagg = makeObject<gcn::GCNNodeLossAggregation>();
            lossagg->setInput(lossjoin);
            Handle<Computation> myWriter = makeObject<gcn::GCNNodeWriter>("gcn", "loss");
            myWriter->setInput(lossagg);

            bool success = pdbClient.executeComputations({myWriter});
        }
        end = std::chrono::steady_clock::now();
        std::cout << "Time for 5 stage: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() << std::endl;
        it = pdbClient.getSetIterator<GCNMatrix>("gcn", "loss");
        count = 0;
        while(it->hasNextRecord()) {
            // grab the record
            auto r = it->getNextRecord();
            count++;

            if(doNotPrint) {
                continue;
            }

            float* values = r->data->c_ptr();
            std::cout << "Loss: " << values[0] << "\n\n\n\n";
            std::cout << "Accuracy: " << values[1]/2708 << "\n\n\n\n";

            std::cout  << "\n\n";
        }


        begin = std::chrono::steady_clock::now();
        {
            const UseTemporaryAllocationBlock tempBlock{1024 * 1024 * 128};

            Handle<Computation> n = makeObject<gcn::GCNNodeScanner>("gcn", "layer2_output");
            Handle<Computation> l = makeObject<gcn::GCNNodeScanner>("gcn", "labels");

            Handle<Computation> Z4GradientJoin = makeObject<gcn::GCNZ4GradientJoin>();
            Z4GradientJoin->setInput(0, n);
            Z4GradientJoin->setInput(1, l);

            Handle<Computation> myWriter = makeObject<gcn::GCNNodeWriter>("gcn", "Z4_gradient");
            myWriter->setInput(Z4GradientJoin);

            bool success = pdbClient.executeComputations({myWriter});
        }
        end = std::chrono::steady_clock::now();
        std::cout << "Time for 6 stage: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() << std::endl;


        begin = std::chrono::steady_clock::now();
        {
            const UseTemporaryAllocationBlock tempBlock{1024 * 1024 * 128};

            Handle<Computation> z3 = makeObject<gcn::GCNNodeScanner>("gcn", "Z3");
            Handle<Computation> z4_gradient = makeObject<gcn::GCNNodeScanner>("gcn", "Z4_gradient");
            Handle<Computation> m2 = makeObject<gcn::GCNMatrixScanner>("gcn", "w2_updated");


            Handle<Computation> W2GradientJoin = makeObject<gcn::GCNW2GradientJoin>();
            W2GradientJoin->setInput(0, z3);
            W2GradientJoin->setInput(1, z4_gradient);
            Handle<Computation> W2GradientAggregation = makeObject<gcn::GCNW2GradientAggregation>();
            W2GradientAggregation->setInput(W2GradientJoin);

            Handle<Computation> W2Updated = makeObject<gcn::GCNMatrixUpdateJoin>();
            W2Updated->setInput(0,W2GradientAggregation);
            W2Updated->setInput(1,m2);

            Handle<Computation> myWriter = makeObject<gcn::GCNMatrixWriter>("gcn", "w2");
            myWriter->setInput(W2Updated);

            bool success = pdbClient.executeComputations({myWriter});
        }
        end = std::chrono::steady_clock::now();

        std::cout << "Time for 7 stage: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() << std::endl;
        begin = std::chrono::steady_clock::now();
        {
            const UseTemporaryAllocationBlock tempBlock{1024 * 1024 * 128};

            Handle<Computation> z4_gradient = makeObject<gcn::GCNNodeScanner>("gcn", "Z4_gradient");
            Handle<Computation> m2 = makeObject<gcn::GCNMatrixScanner>("gcn", "w2_updated");

            Handle<Computation> Z3GradientJoin = makeObject<gcn::GCNZ3GradientJoin>();
            Z3GradientJoin->setInput(0, z4_gradient);
            Z3GradientJoin->setInput(1, m2);

            Handle<Computation> myWriter = makeObject<gcn::GCNNodeWriter>("gcn", "Z3_gradient");
            myWriter->setInput(Z3GradientJoin);

            bool success = pdbClient.executeComputations({myWriter});
        }
        end = std::chrono::steady_clock::now();

        std::cout << "Time for 8 stage: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() << std::endl;
        begin = std::chrono::steady_clock::now();
        {
            const UseTemporaryAllocationBlock tempBlock{1024 * 1024 * 128};


            Handle<Computation> z3_gradient = makeObject<gcn::GCNNodeScanner>("gcn", "Z3_gradient");
            Handle<Computation> e = makeObject<gcn::GCNEdgeScanner>("gcn", "input_edge");


            Handle<Computation> Z2GradientJoin = makeObject<gcn::GCNZ2GradientJoin>();
            Z2GradientJoin->setInput(0,z3_gradient);
            Z2GradientJoin->setInput(1,e);
            Z2GradientJoin->setInput(2,z3_gradient);

            Handle<Computation> Z2Aggregation = makeObject<gcn::GCNNodeAggregation>();
            Z2Aggregation->setInput(Z2GradientJoin);

            Handle<Computation> myWriter = makeObject<gcn::GCNNodeWriter>("gcn", "Z2_gradient");
            myWriter->setInput(Z2Aggregation);

            bool success = pdbClient.executeComputations({myWriter});
        }
        end = std::chrono::steady_clock::now();

        std::cout << "Time for 9 stage: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() << std::endl;
        begin = std::chrono::steady_clock::now();
        {
            const UseTemporaryAllocationBlock tempBlock{1024 * 1024 * 128};

            Handle<Computation> z2_gradient = makeObject<gcn::GCNNodeScanner>("gcn", "Z2_gradient");
            Handle<Computation> z1 = makeObject<gcn::GCNNodeScanner>("gcn", "Z1");
            Handle<Computation> m1 = makeObject<gcn::GCNMatrixScanner>("gcn", "w1_updated");

            Handle<Computation> transformjoin = makeObject<gcn::GCNTransformReLUDerivateJoin>();
            transformjoin->setInput(0,z1);
            transformjoin->setInput(1,m1);

            Handle<Computation> Z1GradientJoin = makeObject<gcn::GCNZ1GradientJoin>();
            Z1GradientJoin->setInput(0,z2_gradient);
            Z1GradientJoin->setInput(1,transformjoin);

            Handle<Computation> myWriter = makeObject<gcn::GCNNodeWriter>("gcn", "Z1_gradient");
            myWriter->setInput(Z1GradientJoin);

            bool success = pdbClient.executeComputations({myWriter});
        }
        end = std::chrono::steady_clock::now();

        std::cout << "Time for 10 stage: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() << std::endl;
        begin = std::chrono::steady_clock::now();
        {
            const UseTemporaryAllocationBlock tempBlock{1024 * 1024 * 128};
            Handle<Computation> z1 = makeObject<gcn::GCNNodeScanner>("gcn", "Z1");
            Handle<Computation> z1_gradient = makeObject<gcn::GCNNodeScanner>("gcn", "Z1_gradient");
            Handle<Computation> m1 = makeObject<gcn::GCNMatrixScanner>("gcn", "w1_updated");

            Handle<Computation> W1GradientJoin = makeObject<gcn::GCNW1GradientJoin>();
            W1GradientJoin->setInput(0, z1);
            W1GradientJoin->setInput(1, z1_gradient);
            Handle<Computation> W1GradientAggregation = makeObject<gcn::GCNW1GradientAggregation>();
            W1GradientAggregation->setInput(W1GradientJoin);

            Handle<Computation> W1Updated = makeObject<gcn::GCNMatrixUpdateJoin>();
            W1Updated->setInput(0, W1GradientAggregation);
            W1Updated->setInput(1, m1);

            Handle<Computation> myWriter = makeObject<gcn::GCNMatrixWriter>("gcn", "w1");
            myWriter->setInput(W1Updated);
            bool success = pdbClient.executeComputations({myWriter});
        }
        end = std::chrono::steady_clock::now();

        std::cout << "Time for 11 stage: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() << std::endl;

        pdbClient.clearSet("gcn", "w1_updated");
        pdbClient.clearSet("gcn", "w2_updated");
        pdbClient.clearSet("gcn","layer1_output");
        pdbClient.clearSet("gcn","layer2_output");
        pdbClient.clearSet("gcn", "loss");

        pdbClient.clearSet("gcn", "Z3");
        pdbClient.clearSet("gcn", "Z1");

        pdbClient.clearSet("gcn", "Z1_gradient");
        pdbClient.clearSet("gcn", "Z2_gradient");
        pdbClient.clearSet("gcn", "Z3_gradient");
        pdbClient.clearSet("gcn", "Z4_gradient");
    }

    auto it = pdbClient.getSetIterator<GCNMatrix>("gcn", "w2");
    int32_t count = 0;
    while(it->hasNextRecord()) {
        // grab the record
        auto r = it->getNextRecord();
        count++;

        // skip if we do not need to print
        if(doNotPrint) {
            continue;
        }
        // write out the values
        // All the results are correct
        std::cout << "number of rows: " << r->numRows << std::endl;
        std::cout << "number of rows: " << r->numCols << std::endl;

        float* values = r->data->c_ptr();
        for(int i = 0; i < r->data->size(); ++i) {
            std::cout << values[i] << ", ";
        }
        std::cout  << "\n\n";
    }
    std::cout << count << '\n';
    // wait a bit before the shutdown
    sleep(4);
    // shutdown the server
    pdbClient.shutDownServer();

    return 0;
}


