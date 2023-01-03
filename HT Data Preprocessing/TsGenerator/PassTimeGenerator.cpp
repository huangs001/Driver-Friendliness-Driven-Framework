#include "PassTimeGenerator.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include "CalTools.h"

void PassTimeGenerator::recordLastInfo(std::size_t threadNo, std::time_t ts, double lat, double lon)
{
	this->lastLatLonAll[threadNo] = std::make_pair(lat, lon);
	this->lastTimeAll[threadNo] = ts;
}

double PassTimeGenerator::calcSpeedInfo(std::size_t threadNo, std::time_t ts, double lat, double lon)
{
	return CalTools::getSpeedFromLatLonInMeterPerSecond(lat, lon, this->lastLatLonAll[threadNo].first, this->lastLatLonAll[threadNo].second, ts - this->lastTimeAll[threadNo]);
}


PassTimeGenerator::PassTimeGenerator(const std::string& csvFolder, const std::string& osmidFolder, const std::string& osmRoadData, const std::string& output) :
	AbstractGenerator(csvFolder, osmidFolder, output),
	osmRoadData(osmRoadData), roadLength(osmRoadData)
{

}

void PassTimeGenerator::start(std::size_t threadNum)
{
	for (std::size_t i = 0; i < threadNum; ++i) {
		this->lastLatLonAll.emplace_back(0, 0);
		this->lastTimeAll.emplace_back(0);
	}
	AbstractGenerator::start(threadNum);
}

DataSaveLoad::InnerType PassTimeGenerator::hit(std::size_t threadNo, unsigned lineIdx, DataLine::IdType osmid, DataLine::IdType taxiId, std::time_t ts, double lon, double lat, const DataSaveLoad::InnerType & oldValue)
{
	if (lineIdx == 0) {
		recordLastInfo(threadNo, ts, lat, lon);
		return oldValue;
	}

	double speed = calcSpeedInfo(threadNo, ts, lat, lon);
	double roadLength = this->roadLength.getRoadLength(osmid);

	if (speed > this->filterSpeed) {
		return oldValue;
	}

	recordLastInfo(threadNo, ts, lat, lon);

	if (roadLength == 0 || speed < 0.1) {
		return oldValue;
	}

	return DataSaveLoad::InnerType(oldValue.first + 1, oldValue.second + (roadLength / speed));
}

void PassTimeGenerator::noHit(std::size_t threadNo, unsigned lineIdx, DataLine::IdType osmid, DataLine::IdType taxiId, std::time_t ts, double lon, double lat, const DataSaveLoad::InnerType& oldValue)
{
	if (lineIdx != 0) {
		double speed = calcSpeedInfo(threadNo, ts, lat, lon);
		if (speed <= this->filterSpeed) {
			recordLastInfo(threadNo, ts, lat, lon);
		}
	}
	else {
		recordLastInfo(threadNo, ts, lat, lon);
	}
}
