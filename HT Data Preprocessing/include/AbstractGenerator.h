#pragma once
#include <string>
#include <filesystem>
#include "DataSaveLoad.h"
#include "ConfigVal.h"
#include "DataLine.h"
#include <unordered_set>

class AbstractGenerator
{
private:
	std::string csvFolder;
	std::string osmidFolder;
	std::string outputPath;
	std::unordered_set<DataLine::IdType> enhanced;

	const unsigned clusterTime = ConfigVal::clusterTime;
public:
	AbstractGenerator(const std::string& csvFolder, const std::string& osmidFolder, const std::unordered_set<DataLine::IdType> &enhanced, const std::string& outputPath);

	virtual void start(std::size_t threadNum);

	virtual void updateFile(std::size_t threadNum, std::size_t fileIdx) = 0;

	virtual DataSaveLoad::InnerType hit(std::size_t threadNum, unsigned lineIdx, DataLine::IdType osmid, DataLine::IdType taxiId, std::time_t ts, double lon, double lat, const DataSaveLoad::InnerType &oldValue) = 0;

	virtual void noHit(std::size_t threadNum, unsigned lineIdx, DataLine::IdType osmid, DataLine::IdType taxiId, std::time_t ts, double lon, double lat, const DataSaveLoad::InnerType& oldValue) = 0;

	virtual ~AbstractGenerator() {}
};

