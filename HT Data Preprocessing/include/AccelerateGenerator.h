#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <ctime>
#include "AbstractGenerator.h"
#include "ConfigVal.h"
#include "RoadLength.h"

class AccelerateGenerator : public AbstractGenerator
{
private:
	std::string osmRoadData;
	RoadLength roadLength;
	std::vector<unsigned> lastLineAll;
	std::vector<std::pair<double, double>> lastLatLonAll;
	std::vector<double> lastSpeedAll;
	std::vector<std::time_t> lastTimeAll;

	void recordLastInfo(std::size_t threadNo, std::time_t ts, double lat, double lon);
	double calcSpeedInfo(std::size_t threadNo, std::time_t ts, double lat, double lon) const;
	inline static const double filterSpeed = ConfigVal::filterSpeed;
public:
	AccelerateGenerator(const std::string& csvFolder, const std::string& osmidFolder, const std::string& osmRoadData, const std::string& output);

	void start(std::size_t threadNum) override;

	void updateFile(std::size_t threadNo, std::size_t newName) override {};

	DataSaveLoad::InnerType hit(std::size_t threadNo, unsigned lineIdx, DataLine::IdType osmid, DataLine::IdType taxiId, std::time_t ts, double lon, double lat, const DataSaveLoad::InnerType& oldValue) override;

	void noHit(std::size_t threadNo, unsigned lineIdx, DataLine::IdType osmid, DataLine::IdType taxiId, std::time_t ts, double lon, double lat, const DataSaveLoad::InnerType& oldValue) override;
};

