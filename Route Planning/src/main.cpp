#include <iostream>
#include <random>
#include <vector>
#include <fstream>
#include <sstream>
#include <map>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include "routingkit/all.h"
#include "routingkit/friend_weight.h"
#include "args.hxx"

struct Road_type {
	unsigned small_id;
	unsigned big_id;
	unsigned u;
	unsigned v;
	double length;
	double travel_time;
	double blur_time;
	double safety;
	double comfort;
	
	bool operator<(const Road_type &rhs) const {
		return small_id < rhs.small_id;
	}
};

std::unordered_map<unsigned, Road_type> load_weight(const std::string &path) {
	std::unordered_map<unsigned, Road_type> ret;
	std::ifstream ifs(path);
	std::string line;
	std::string cell;
	unsigned id1, id2, u, v;
	double val1, val2, val3, val4, val5;
	while (std::getline(ifs, line)) {
		std::stringstream ss(line);
		std::vector<std::string> split;
		while (std::getline(ss, cell, ',')) {
			split.push_back(cell);
		}
		id1 = std::stoul(split[0]);
		id2 = std::stoul(split[1]);
		u = std::stoul(split[2]);
		v = std::stoul(split[3]);
		val1 = std::stod(split[4]);
		val2 = std::stod(split[5]);
		val3 = std::stod(split[6]);
		val4 = std::stod(split[7]);
		val5 = std::stod(split[8]);

		ret[id1] = {id1, id2, u, v, val1, val2, val3, val4, val5};
	}

	return ret;
}

int main(int argc, char* argv[]) {
	args::ArgumentParser parser("cch kernal friendly routing demo", "");
    args::HelpFlag help(parser, "help", "display this help menu", {'h', "help"});
    args::ValueFlag<std::string> weightArgs(parser, "path", "weight graph file", {'w'});
    args::ValueFlag<unsigned> oArgs(parser, "origin", "origin point", {'o'});
    args::ValueFlag<unsigned> dArgs(parser, "dest", "destination point", {'d'});

    if (argc <= 1) {
        std::cerr << parser;
        return 0;
    }

    try {
        parser.ParseCLI(argc, argv);
    } catch (const args::Help&) {
        std::cerr << parser;
        return 0;
    } catch (const args::ParseError &e) {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    }
	

	auto weight_path = args::get(weightArgs);
	auto from = args::get(oArgs);
	auto to = args::get(dArgs);

	auto mysuper_weight = load_weight(weight_path);

	std::vector<unsigned> tail;
	std::vector<unsigned> head;
	std::unordered_set<unsigned> id_set;
	std::unordered_map<unsigned, unsigned> id_map1;
	std::unordered_map<unsigned, unsigned> id_map2;

	for (auto &ww : mysuper_weight) {
		id_set.insert(ww.second.u);
		id_set.insert(ww.second.v);
	}
	std::cout << id_set.size() << std::endl;
	//return 0;

	int idx = 0;
	for (auto &wwi : id_set) {
		id_map1[idx] = wwi;
		id_map2[wwi] = idx;
		++idx;
	}

	std::map<std::pair<unsigned, unsigned>, Road_type> id_map3;

	for (auto &ww : mysuper_weight) {
		id_map3[std::make_pair(id_map2[ww.second.u], id_map2[ww.second.v])] = ww.second;
		tail.push_back(id_map2[ww.second.u]);
		head.push_back(id_map2[ww.second.v]);
	}


	// Build the shortest path index
	std::vector<RoutingKit::friend_weight> fw;

	for (auto &www : mysuper_weight) {
		auto &ww = www.second;
		fw.emplace_back(unsigned(ww.blur_time * 1000), unsigned(ww.safety * 100), unsigned(ww.comfort * 100), unsigned(ww.length * 100));
	}

	std::vector<float> latitude;
	std::vector<float> longitude;

	for (int i = 0; i < mysuper_weight.size(); ++i) {
		latitude.push_back(115.7);
		longitude.push_back(39.4);
	}

	std::vector<unsigned>node_order = RoutingKit::compute_nested_node_dissection_order_using_inertial_flow(mysuper_weight.size(), tail, head, latitude, longitude);

	RoutingKit::CustomizableContractionHierarchy cch(node_order, tail, head);

	RoutingKit::CustomizableContractionHierarchyMetric metric(cch, fw);
	metric.customize();
	RoutingKit::CustomizableContractionHierarchyQuery cch_query(metric);

	cch_query.reset().add_source(from).add_target(to).run();

	auto d2 = cch_query.get_distance();
	auto path2 = cch_query.get_node_path();

	std::cout << "To get from "<< from << " to "<< to << " one needs " << d2.length << " milliseconds." << std::endl;
	std::cout << "The path is";
	for(auto x:path2)
		std::cout << " " << x;
	std::cout << std::endl;

    return 0;
}