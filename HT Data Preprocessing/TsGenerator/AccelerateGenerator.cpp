#include "AccelerateGenerator.h"
#include <fstream>
#include <sstream>
#include <cmath>
#include <iostream>
#include "CalTools.h"
#include "ConfigVal.h"

void AccelerateGenerator::recordLastInfo(std::size_t threadNo, std::time_t ts, double lat, double lon)
{
	this->lastLatLonAll[threadNo] = std::make_pair(lat, lon);
	this->lastTimeAll[threadNo] = ts;
}

double AccelerateGenerator::calcSpeedInfo(std::size_t threadNo, std::time_t ts, double lat, double lon) const
{
	return CalTools::getSpeedFromLatLonInMeterPerSecond(lat, lon, this->lastLatLonAll[threadNo].first, this->lastLatLonAll[threadNo].second, ts - this->lastTimeAll[threadNo]);
}

AccelerateGenerator::AccelerateGenerator(const std::string& csvFolder, const std::string& osmidFolder, const std::string& osmRoadData, const std::string& output) :
	AbstractGenerator(csvFolder, osmidFolder, output),
	osmRoadData(osmRoadData), roadLength(osmRoadData)
{

}

void AccelerateGenerator::start(std::size_t threadNum)
{
	for (std::size_t i = 0; i < threadNum; ++i) {
		this->lastLatLonAll.emplace_back(0, 0);
		this->lastTimeAll.emplace_back(0);
		this->lastSpeedAll.emplace_back(0);
	}
	AbstractGenerator::start(threadNum);
}

DataSaveLoad::InnerType AccelerateGenerator::hit(std::size_t threadNo, unsigned lineIdx, DataLine::IdType osmid, DataLine::IdType taxiId, std::time_t ts, double lon, double lat, const DataSaveLoad::InnerType& oldValue)
{
	if (lineIdx == 0) {
		recordLastInfo(threadNo, ts, lat, lon);
		return oldValue;
	}

	double speed = calcSpeedInfo(threadNo, ts, lat, lon);

	if (speed > this->filterSpeed) {
		return oldValue;
	}

	double& lastSpeed = this->lastSpeedAll[threadNo];
	auto lastTime = this->lastTimeAll[threadNo];

	recordLastInfo(threadNo, ts, lat, lon);

	if (lineIdx == 1) {
		lastSpeed = speed;
		return oldValue;
	}

	auto tmpLastSpeed = lastSpeed;
	lastSpeed = speed;

	return DataSaveLoad::InnerType(oldValue.first + 1, oldValue.second + (std::abs(tmpLastSpeed - speed) / (ts - lastTime)));
}

void AccelerateGenerator::noHit(std::size_t threadNo, unsigned lineIdx, DataLine::IdType osmid, DataLine::IdType taxiId, std::time_t ts, double lon, double lat, const DataSaveLoad::InnerType& oldValue)
{
	if (lineIdx != 0) {
		double speed = calcSpeedInfo(threadNo, ts, lat, lon);
		if (speed <= this->filterSpeed) {
			this->lastSpeedAll[threadNo] = speed;
			recordLastInfo(threadNo, ts, lat, lon);
		}
	}
	else {
		recordLastInfo(threadNo, ts, lat, lon);
	}
}

