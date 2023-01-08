#include "DataParser.h"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include "SliceMultiThread.h"
#include "DataLine.h"

namespace fs = std::filesystem;

DataParser::DataParser(std::filesystem::path csvFolder, std::filesystem::path osmidFolder) :
	csvFolder(std::move(csvFolder)),
	osmidFolder(std::move(osmidFolder))
{	
	for (const auto& entry : fs::directory_iterator(this->csvFolder)) {
		filePaths.emplace_back(entry.path().filename());
	}

	std::sort(filePaths.begin(), filePaths.end(), [](const fs::path &f1, const fs::path &f2) {
		return std::stoul(f1.stem()) < std::stoul(f2.stem());
	});
}

void DataParser::parse(std::function<void(std::size_t, std::size_t, const std::vector<DataLine> &)> fun, unsigned threadNum)
{
	SliceMultiThread::multiThread(filePaths, threadNum, [&](unsigned threadNo, std::size_t idx, const fs::path &which) {
		auto data = this->parse(idx + 1);
		fun(threadNo, idx, data);
	});
}

std::size_t DataParser::size()
{
	return filePaths.size();
}

std::vector<DataLine> DataParser::parse(std::size_t fileNum)
{
	std::vector<DataLine> result;
	auto path1 = DataParser::csvFolder / filePaths[fileNum - 1];
	auto path2 = DataParser::osmidFolder / filePaths[fileNum - 1];
	std::ifstream if1(path1), if2(path2);
	std::string line1, line2;

	unsigned lineCnt = 0;

	while (std::getline(if1, line1) && std::getline(if2, line2)) {
		std::stringstream ss1(line1);
		std::string id, timeStamp, lon, lat;
		std::getline(ss1, id, ',');
		std::getline(ss1, timeStamp, ',');
		std::getline(ss1, lon, ',');
		std::getline(ss1, lat, ',');
		std::tm tm = {};
		std::stringstream tss(timeStamp);
		tss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");
		auto tp = std::mktime(&tm);

		std::vector<DataLine::IdType> osmidList;
		std::stringstream ss2(line2);
		while (std::getline(ss2, line2, ',')) {
			osmidList.emplace_back(DataLine::castid(line2));
		}

		result.emplace_back(DataLine::castid(id), tp, std::stod(lon), std::stod(lat), osmidList);
	}

	return result;
}

