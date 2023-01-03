#include <iostream>
#include <string>
#include "FlowGenerator.h"
#include "DataSaveLoad.h"
#include "CalTools.h"
#include "PassTimeGenerator.h"
#include "AccelerateGenerator.h"
#include <fstream>
#include <cmath>
#include <algorithm>
#include <random>
#include "TrajClassify.h"
#include "AdjGraphFilter.h"
#include "RoadLength.h"
#include "args.hxx"


void testCalc()
{
	std::cout << CalTools::getDistanceFromLatLonInKm(39.92123, 116.51172, 39.93883, 116.51135) << std::endl;
}

void generateData(ConfigVal config) {
	FlowGenerator fg(config.csv_folder, config.osmid_folder, config.osmDataPath, config.bigFlowOutput);
	fg.start(2);


	std::cout << "Step 1" << std::endl;
	PassTimeGenerator ptg(config.csv_folder, config.osmid_folder, config.osmDataPath, config.bigPtOutput);
	ptg.start(2);
	std::cout << "Step 2" << std::endl;

	AccelerateGenerator ag(config.csv_folder, config.osmid_folder, config.osmDataPath, config.bigAccOutput);
	ag.start(2);
	std::cout << "End" << std::endl;
}


void trjCls(ConfigVal config) {
	AdjGraphFilter adjFilter(std::filesystem::path(config.csv_folder), std::filesystem::path(config.osmDataPath), std::filesystem::path(config.smallNodePath), std::filesystem::path(config.smallEdgePath));

	TrajClassify tjc(config.csv_folder, config.osmid_folder, config.osmDataPath, config.bigFlowOutput, config.trjClsOutDir);
	std::cout << "start" << std::endl;
	auto edgeSet = adjFilter.edgeOsmIdSet();
	const unsigned limit = 5;
	tjc.start(4, [&](const std::vector<DataLine> &data) {
		unsigned cnt = 0;
		for (auto &line : data) {
			if (edgeSet.count(line.osmid[0]) > 0 && ++cnt >= limit) {
				break;
			}
		}
		return cnt >= limit;
	});
}

void adjProcess(ConfigVal config) {
	AdjGraphFilter adjFilter(std::filesystem::path(config.csv_folder), std::filesystem::path(config.osmid_folder), config.smallNodePath, config.smallEdgePath);
	adjFilter.startFilter(8, 300,
		{
					config.trjClsOutDir / std::filesystem::path("flow.txt"),
					config.trjClsOutDir / std::filesystem::path("passtime.txt") ,
					config.trjClsOutDir / std::filesystem::path("accelerate.txt") }, config.filterClsOutDir,
		{ "flow.txt", "passtime.txt", "accelerate.txt" });
}

void getPlainData(ConfigVal config) {
	RoadLength roadLengthGetter(config.osmDataPath);
	AdjGraphFilter adjFilter(std::filesystem::path(config.csv_folder), std::filesystem::path(config.osmid_folder), config.smallNodePath, config.smallEdgePath);
	auto edgeSet = adjFilter.edgeOsmIdSet();

	auto flowData = DataSaveLoad::loadData(config.bigFlowOutput);
	
	DataSaveLoad::dumpData(flowData, config.smallFlowOutput, [&](DataSaveLoad::IdType id, const std::vector<DataSaveLoad::InnerType> &) { return edgeSet.count(id) > 0; });
	auto flowFilterDataTmp = DataSaveLoad::loadData(config.smallFlowOutput);

	std::vector<std::tuple<double, DataLine::IdType>> pre;

	for (auto& item : flowFilterDataTmp) {
		std::size_t cnt = 0;
		for (auto& val : item.second) {
			if (val.first > 0) {
				++cnt;
			}
		}
		pre.emplace_back(1.0 * cnt / item.second.size(), item.first);
	}

	std::sort(pre.begin(), pre.end());
	std::reverse(pre.begin(), pre.end());
	int limit = config.limit;
	edgeSet.clear();
	for (auto val : pre) {
		edgeSet.emplace(std::get<1>(val));
		std::cout << std::get<0>(val) << " ";
		if (--limit == 0) {
			break;
		}
	}
	std::cout << std::endl;

	std::default_random_engine e(233);
	std::uniform_int_distribution<int> u(0, 2);

	std::cout << "1" << std::endl;
	DataSaveLoad::dumpData(flowData, config.smallFlowOutput, [&](DataSaveLoad::IdType id, const std::vector<DataSaveLoad::InnerType>&) { return edgeSet.count(id) > 0; });
	auto flowFilterData = DataSaveLoad::loadData(config.smallFlowOutput);
	DataSaveLoad::dumpPlainData(flowFilterData, config.smallPlainFlowOutput, [&](DataSaveLoad::IdType id, DataSaveLoad::CountType cnt, DataSaveLoad::ValueType val) {
		return cnt + 1;
	});
	
	std::cout << "2" << std::endl;
	auto ptData = DataSaveLoad::loadData(config.bigPtOutput);
	DataSaveLoad::dumpData(ptData, config.smallPtOutput, [&](DataSaveLoad::IdType id, const std::vector<DataSaveLoad::InnerType> &) { return edgeSet.count(id) > 0; });
	auto ptFilterData = DataSaveLoad::loadData(config.smallPtOutput);
	DataSaveLoad::dumpPlainData(ptFilterData, config.smallPlainPtOutput, [&](DataSaveLoad::IdType id, DataSaveLoad::CountType cnt, DataSaveLoad::ValueType val) {
		double pval = 0;
		if (cnt > 0) {
			//pval = std::floor(val / cnt / 100);
			pval = val / cnt;
			//pval = std::min(15.0, pval);
			/*if (val / cnt / 100 < 1 && val / cnt > 0.5) {
				pval = 1;
			}*/
			if (val / cnt < 1) {
				pval = 1;
			}
		} else {
			auto roadLength = roadLengthGetter.getRoadLength(id);
			pval = roadLength / 2;
		}
		return std::max(1, static_cast<int>(std::floor(pval / 100)));
	});

	std::cout << "3" << std::endl;
	auto accData = DataSaveLoad::loadData(config.bigAccOutput);
	DataSaveLoad::dumpData(accData, config.smallAccOutput, [&](DataSaveLoad::IdType id, const std::vector<DataSaveLoad::InnerType> &) { return edgeSet.count(id) > 0; });
	auto accFilterData = DataSaveLoad::loadData(config.smallAccOutput);
	DataSaveLoad::dumpPlainData(accFilterData, config.smallPlainAccOutput, [&](DataSaveLoad::IdType id, DataSaveLoad::CountType cnt, DataSaveLoad::ValueType val) {
		double pval = 0;
		if (cnt > 0) {
			//pval = std::floor(val / cnt * 50);
			pval = val / cnt * 100;
			//pval = std::min(15.0, pval);
			if (pval < 1) {
				pval = 1;
			}
		} else {
			pval = 1;
		}
		return std::max(1, static_cast<int>(std::round(pval)));
	});
	std::cout << "4" << std::endl;
}

void getPlainData2(ConfigVal config) {
	RoadLength roadLengthGetter(config.osmDataPath);
	AdjGraphFilter adjFilter(std::filesystem::path(config.csv_folder), std::filesystem::path(config.osmid_folder), config.smallNodePath, config.smallEdgePath);
	auto edgeSet = adjFilter.edgeOsmIdSet();

	auto flowData = DataSaveLoad::loadData(config.bigFlowOutput);

	DataSaveLoad::dumpData(flowData, config.smallFlowOutput2, [&](DataSaveLoad::IdType id, const std::vector<DataSaveLoad::InnerType>&) { return edgeSet.count(id) > 0; });
	auto flowFilterData = DataSaveLoad::loadData(config.smallFlowOutput);
	DataSaveLoad::dumpPlainData(flowFilterData, config.smallPlainFlowOutput2, [&](DataSaveLoad::IdType id, DataSaveLoad::CountType cnt, DataSaveLoad::ValueType val) {
		//int ival = cnt;
		//if (ival != 0) {
		//	auto luck = u(e);
		//	ival = 3 * ival - luck;
		//} else {
		//	ival = 1;
		//}
		//return ival;
		//double pval = cnt + (val - cnt) / cnt / 1.7;
		//return std::max(1, static_cast<int>(std::round(pval)));
		return cnt + 1;
		});

	std::cout << "2" << std::endl;
	auto ptData = DataSaveLoad::loadData(config.bigPtOutput);
	DataSaveLoad::dumpData(ptData, config.smallPtOutput2, [&](DataSaveLoad::IdType id, const std::vector<DataSaveLoad::InnerType>&) { return edgeSet.count(id) > 0; });
	auto ptFilterData = DataSaveLoad::loadData(config.smallPtOutput);
	DataSaveLoad::dumpPlainData(ptFilterData, config.smallPlainPtOutput2, [&](DataSaveLoad::IdType id, DataSaveLoad::CountType cnt, DataSaveLoad::ValueType val) {
		double pval = 0;
		if (cnt > 0) {
			//pval = std::floor(val / cnt / 100);
			pval = val / cnt;
			//pval = std::min(15.0, pval);
			/*if (val / cnt / 100 < 1 && val / cnt > 0.5) {
				pval = 1;
			}*/
			if (val / cnt < 1) {
				pval = 1;
			}
		}
		else {
			auto roadLength = roadLengthGetter.getRoadLength(id);
			pval = roadLength / 2;
		}
		return std::max(1, static_cast<int>(std::floor(pval / 100)));
		});



	std::cout << "3" << std::endl;
	auto accData = DataSaveLoad::loadData(config.bigAccOutput);
	DataSaveLoad::dumpData(accData, config.smallAccOutput2, [&](DataSaveLoad::IdType id, const std::vector<DataSaveLoad::InnerType>&) { return edgeSet.count(id) > 0; });
	auto accFilterData = DataSaveLoad::loadData(config.smallAccOutput);
	DataSaveLoad::dumpPlainData(accFilterData, config.smallPlainAccOutput2, [&](DataSaveLoad::IdType id, DataSaveLoad::CountType cnt, DataSaveLoad::ValueType val) {
		double pval = 0;
		if (cnt > 0) {
			//pval = std::floor(val / cnt * 50);
			pval = val / cnt * 100;
			//pval = std::min(15.0, pval);
			if (pval < 1) {
				pval = 1;
			}
		}
		else {
			pval = 1;
		}
		return std::max(1, static_cast<int>(std::round(pval)));
		});
	std::cout << "4" << std::endl;
}

void env(ConfigVal config) {
	//checkAvailable(config.smallIdPath);
	//generateData(config);
	trjCls(config);
	adjProcess(config);
	getPlainData(config);
}

int main(int argc, char *argv[])
{
	args::ArgumentParser parser("HT data processing", "");
	args::HelpFlag help(parser, "help", "display this help menu", { 'h', "help" });
	args::ValueFlag<std::string> arg1(parser, "csv_folder", "csv_file", { "csv" });
	args::ValueFlag<std::string> arg2(parser, "osmid_folder", "osmid_folder", { "osmid" });
	args::ValueFlag<std::string> arg3(parser, "osmDataPath", "osmDataPath", { "osmdata" });
	args::ValueFlag<std::string> arg4(parser, "partEdgePath", "partEdgePath", { "part_edge" });
	args::ValueFlag<std::string> arg5(parser, "partNodePath", "partNodePath", { "part_node" });
	args::ValueFlag<std::string> arg6(parser, "partIdPath", "partIdPath", { "part_path" });

	if (argc <= 1) {
		std::cerr << parser;
		return 0;
	}

	try {
		parser.ParseCLI(argc, argv);
	}
	catch (const args::Help&) {
		std::cerr << parser;
		return 0;
	}
	catch (const args::ParseError& e) {
		std::cerr << e.what() << std::endl;
		std::cerr << parser;
		return 1;
	}

	auto arg1_val = args::get(arg1);
	auto arg2_val = args::get(arg2);
	auto arg3_val = args::get(arg3);
	auto arg4_val = args::get(arg4);
	auto arg5_val = args::get(arg5);
	auto arg6_val = args::get(arg6);

	ConfigVal config1 = ConfigVal::getConfigSmall1();

	config1.csv_folder = arg1_val;
	config1.osmid_folder = arg2_val;
	config1.osmDataPath = arg3_val;
	config1.smallEdgePath = arg4_val;
	config1.smallNodePath = arg5_val;
	config1.smallIdPath = arg6_val;

	generateData(config1);
	trjCls(config1);
	adjProcess(config1);
	getPlainData(config1);

	return 0;
}