#include "FlowGenerator.h"
#include <filesystem>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <thread>
#include <fstream>
#include <sstream>
#include <ctime>
#include <set>
#include <mutex>
#include "DataParser.h"
#include "DataSaveLoad.h"
#include "ConfigVal.h"

FlowGenerator::FlowGenerator(const std::string& csvFolder, const std::string& osmidFolder, const std::string& osmRoadData, const std::unordered_set<DataLine::IdType> &enhanced, const std::string& output) :
	AbstractGenerator(csvFolder, osmidFolder, enhanced, output), roadLength(osmRoadData) {}

DataSaveLoad::InnerType FlowGenerator::hit(std::size_t threadNo, unsigned lineIdx, DataLine::IdType osmid, DataLine::IdType taxiId, std::time_t ts, double lon, double lat, const DataSaveLoad::InnerType& oldValue)
{
	auto length = roadLength.getRoadLength(osmid);
	double adjust = 1.0;
	if (length > 0) {
		adjust = length / 200.0;
	}
	return DataSaveLoad::InnerType(oldValue.first + 1, oldValue.second + adjust);
}



