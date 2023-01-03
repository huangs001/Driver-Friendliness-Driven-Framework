#include "AbstractGenerator.h"
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
#include "DataLine.h"


AbstractGenerator::AbstractGenerator(const std::string& csvFolder, const std::string& osmidFolder, const std::string& outputPath) :
	csvFolder(csvFolder),
	osmidFolder(osmidFolder),
	outputPath(outputPath)
{}

void AbstractGenerator::start(std::size_t threadNum)
{
	namespace fs = std::filesystem;

	//std::stringstream sss("2008-2-2 00:00:00"), sse("2008-2-9 00:00:00");
	std::stringstream sss("2008-2-2 00:00:00"), sse("2008-2-16 00:00:00");
	
	std::tm tm1 = {}, tm2 = {};
	sss >> std::get_time(&tm1, "%Y-%m-%d %H:%M:%S");
	sse >> std::get_time(&tm2, "%Y-%m-%d %H:%M:%S");
	auto tps = std::mktime(&tm1);
	auto tpe = std::mktime(&tm2);
	std::cout << (tpe - tps + clusterTime) / clusterTime << std::endl;

	std::vector<DataSaveLoad::TSDataType> osmDataAll;

	std::vector<std::size_t> cntAll;

	std::mutex lock;

	DataParser dataParser(this->csvFolder, this->osmidFolder);

	for (std::size_t i = 0; i < threadNum; ++i) {
		osmDataAll.emplace_back();
		cntAll.emplace_back(0);
	}



	dataParser.parse([&](std::size_t threadNo, std::size_t fileIdx, const std::vector<DataLine> &data) {
		auto& partOsmData = osmDataAll[threadNo];
		std::set<std::pair<std::size_t, std::time_t>> chkSet;
		std::time_t lastTime = 0;
		auto allCnt = (dataParser.size() - threadNo) / threadNum;

		this->updateFile(threadNo, fileIdx);

		for (std::size_t i = 0; i < data.size(); ++i) {
			auto &line = data[i];
			auto lineIdx = i;
			auto tp = line.ts;
			auto &osmidList = line.osmid;
			auto taxiId = line.taxiID;
			auto lon = line.lon;
			auto lat = line.lat;
			if (tp != lastTime) {
				//for (const auto& osmid : osmidList) {
				//	partOsmData.try_emplace(osmid, (tpe - tps + clusterTime) / clusterTime);
				//	auto timeIdx = (tp - tps) / clusterTime;
				//	if (chkSet.count({ osmid, timeIdx }) == 0) {
				//		//partOsmData[osmid][timeIdx] += 1;
				//		partOsmData[osmid][timeIdx] = this->hit(threadNo, lineIdx, osmid, taxiId, tp, lon, lat, partOsmData[osmid][timeIdx]);
				//		chkSet.emplace(osmid, timeIdx);
				//	}
				//	else {
				//		this->noHit(threadNo, lineIdx, osmid, taxiId, tp, lon, lat, partOsmData[osmid][timeIdx]);
				//	}
				//}

				auto osmid = osmidList.front();
				
				partOsmData.try_emplace(osmid, (tpe - tps + clusterTime) / clusterTime);
				auto timeIdx = (tp - tps) / clusterTime;
				auto &part = partOsmData[osmid];
				if (timeIdx < part.size()) {
					if (chkSet.count({ osmid, timeIdx }) == 0) {
						//partOsmData[osmid][timeIdx] += 1;
						part[timeIdx] = this->hit(threadNo, lineIdx, osmid, taxiId, tp, lon, lat, part[timeIdx]);
						chkSet.emplace(osmid, timeIdx);
					}
					else {
						this->noHit(threadNo, lineIdx, osmid, taxiId, tp, lon, lat, part[timeIdx]);
					}
					lastTime = tp;
				}
			}
		}

		if (++cntAll[threadNo] % 50 == 0) {
			{
				std::lock_guard<std::mutex> guard(lock);
				std::cout << "Thread" << threadNo << ": " << cntAll[threadNo] << "/" << allCnt << std::endl;
			}
		}
		}, threadNum);

	DataSaveLoad::TSDataType osmData;

	for (auto &data : osmDataAll) {
		for (const auto& kv : data) {
			osmData.try_emplace(kv.first, (tpe - tps + clusterTime) / clusterTime);
			auto& tmp = osmData[kv.first];
			for (std::size_t i = 0; i < kv.second.size(); ++i) {
				tmp[i].first += kv.second[i].first;
				tmp[i].second += kv.second[i].second;
			}
		}
		//data.clear();
	}

	DataSaveLoad::dumpData(osmData, outputPath);
}
