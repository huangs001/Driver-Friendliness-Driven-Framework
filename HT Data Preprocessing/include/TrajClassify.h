#pragma once
#include <functional>
#include <tuple>
#include <string>
#include <vector>
#include <unordered_map>
#include "DataSaveLoad.h"
#include "DataLine.h"

class TrajClassify
{
private:
	std::unordered_map<DataLine::IdType, double> roadLength;
	DataSaveLoad::TSDataType flowTsData;
	std::string csvFolder;
	std::string osmidFolder;
	std::string outputPath;
	std::tuple<double, double, double> getAvg(const std::vector<unsigned> &flowCnt, const std::vector<double>& ptCnt, const std::vector<double>& accCnt);
public:
	TrajClassify(const std::string& csvFolder, const std::string& osmidFolder, const std::string &osmRoadData, const std::string &flowTsData, const std::string& outputPath);
	void start(unsigned threadNum);
	void start(unsigned threadNum, std::function<bool(const std::vector<DataLine> &)> filter);
};

