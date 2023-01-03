#include "TrajClassify.h"
#include <filesystem>
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <iostream>
#include <fstream>
#include <sstream>
#include "DataParser.h"
#include "CalTools.h"
#include "ConfigVal.h"
#include "DataSaveLoad.h"
#include "DataLine.h"

std::tuple<double, double, double> TrajClassify::getAvg(const std::vector<unsigned>& flowCnt, const std::vector<double>& ptCnt, const std::vector<double>& accCnt)
{
	double results[3] = {0, 0, 0};

	if (!flowCnt.empty()) {
		for (auto& f : flowCnt) {
			results[0] += f;
		}
		results[0] /= flowCnt.size();
	}

	if (!ptCnt.empty()) {
		for (auto& f : ptCnt) {
			results[1] += f;
		}
		results[1] /= ptCnt.size();
	}

	if (!accCnt.empty()) {
		for (auto& f : accCnt) {
			results[2] += f;
		}
		results[2] /= accCnt.size();
	}

	return std::tuple<double, double, double>(results[0], results[1], results[2]);
}


TrajClassify::TrajClassify(const std::string& csvFolder, const std::string& osmidFolder, const std::string& osmRoadData, const std::string& flowTsData, const std::string& outputPath) :
	csvFolder(csvFolder),
	osmidFolder(osmidFolder),
	outputPath(outputPath)
{
	std::ifstream ifs(osmRoadData);
	std::string line;

	while (std::getline(ifs, line)) {
		std::stringstream ss(line);
		std::string id, dist;
		std::getline(ss, id, ',');
		std::getline(ss, dist, ',');

		this->roadLength.emplace(DataLine::castid(id), std::stod(dist));
	}

	this->flowTsData = DataSaveLoad::loadData(flowTsData);
}

void TrajClassify::start(unsigned threadNum)
{
	start(threadNum, [](const std::vector<DataLine> &) { return true;  });
}

void TrajClassify::start(unsigned threadNum, std::function<bool(const std::vector<DataLine>&)> filter) {
	namespace fs = std::filesystem;

	std::stringstream sss("2008-2-2 00:00:00"), sse("2008-2-16 00:00:00");
	std::tm tm1 = {}, tm2 = {};
	sss >> std::get_time(&tm1, "%Y-%m-%d %H:%M:%S");
	sse >> std::get_time(&tm2, "%Y-%m-%d %H:%M:%S");
	auto tps = std::mktime(&tm1);
	auto tpe = std::mktime(&tm2);
	auto clusterTime = ConfigVal::clusterTime;


	std::vector<std::size_t> cntAll;

	std::vector < std::tuple < std::vector<std::pair<double, unsigned>>, std::vector<std::pair<double, unsigned>>, std::vector<std::pair<double, unsigned>>>> storeAll;

	std::vector<std::thread> threads;

	std::mutex lock;

	DataParser dataParser(this->csvFolder, this->osmidFolder);

	for (std::size_t i = 0; i < threadNum; ++i) {
		cntAll.emplace_back(0);
		storeAll.emplace_back();
	}

	auto dealFun = [&](const std::vector<unsigned> &f, const std::vector<double> &p, const std::vector<double> &a, DataLine::IdType tid, std::tuple < std::vector<std::pair<double, unsigned>>, std::vector<std::pair<double, unsigned>>, std::vector<std::pair<double, unsigned>>> &store) {
		auto result = getAvg(f, p, a);
		std::get<0>(store).emplace_back(std::get<0>(result), tid);
		std::get<1>(store).emplace_back(std::get<1>(result), tid);
		std::get<2>(store).emplace_back(std::get<2>(result), tid);
	};


	dataParser.parse([&](std::size_t threadNo, std::size_t fileIdx, const std::vector<DataLine> &data) {
		if (filter(data)) {

			std::vector<unsigned> flowCnt;
			std::vector<double> ptCnt;
			std::vector<double> accCnt;

			auto &store = storeAll[threadNo];
			std::time_t lastTime = 0;
			std::time_t trueLastTime = 0;
			std::pair<double, double> lastLatLon;
			double lastSpeed = 0;

			for (std::size_t i = 0; i < data.size(); ++i) {
				bool record = true;
				auto &line = data[i];
				auto tp = line.ts;
				auto lineIdx = i;
				auto &osmidList = data[i].osmid;
				auto lat = line.lat;
				auto lon = line.lon;
				if (tp != trueLastTime) {
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

					if (lineIdx > 0) {
						double dist = CalTools::getDistanceFromLatLonInKm(lat, lon, lastLatLon.first, lastLatLon.second);
						double speed = CalTools::getSpeedFromLatLonInMeterPerSecond(lat, lon, lastLatLon.first, lastLatLon.second, tp - lastTime);
						double roadLength = this->roadLength[osmid];

						if (speed <= ConfigVal::filterSpeed) {
							lastLatLon = std::make_pair(lat, lon);

							if (roadLength > 0) {
								ptCnt.emplace_back(roadLength / speed);
							}
							if (lineIdx > 1) {
								accCnt.emplace_back(std::abs(lastSpeed - speed) / (tp - lastTime));
							}
							lastSpeed = speed;
						}
						else {
							record = false;
						}
					}

					if (record) {
						lastTime = tp;
					}
					trueLastTime = tp;

					auto timeIdx = (tp - tps) / clusterTime;
					auto &osmTs = this->flowTsData[osmid];
					auto flow = osmTs[timeIdx].first;
					flowCnt.emplace_back(flow);

				}
			}

			if (!flowCnt.empty() || !ptCnt.empty() || !accCnt.empty()) {
				dealFun(flowCnt, ptCnt, accCnt, data.back().taxiID, store);
			}

			
		}
		auto allCnt = (dataParser.size() - threadNo) / threadNum;
		if (++cntAll[threadNo] % 50 == 0) {
			{
				std::lock_guard<std::mutex> guard(lock);
				std::cout << "Thread" << threadNo << ": " << cntAll[threadNo] << "/" << allCnt << std::endl;
			}
		}
	}, threadNum);

	for (auto &thread : threads) {
		thread.join();
	}

	// merge
	std::vector<std::pair<double, unsigned>> flowSort, ptSort, accSort;

	for (std::size_t threadNo = 0; threadNo < threadNum; ++threadNo) {
		for (std::size_t i = 0; i < std::get<0>(storeAll[threadNo]).size(); ++i) {
			flowSort.emplace_back(std::get<0>(storeAll[threadNo])[i]);
		}

		for (std::size_t i = 0; i < std::get<1>(storeAll[threadNo]).size(); ++i) {
			ptSort.emplace_back(std::get<1>(storeAll[threadNo])[i]);
		}

		for (std::size_t i = 0; i < std::get<2>(storeAll[threadNo]).size(); ++i) {
			accSort.emplace_back(std::get<2>(storeAll[threadNo])[i]);
		}
	}

	std::sort(flowSort.begin(), flowSort.end());
	std::sort(ptSort.begin(), ptSort.end());
	std::sort(accSort.begin(), accSort.end());

	std::ofstream o1(this->outputPath / fs::path("flow.txt"));

	for (auto &f : flowSort) {
		o1 << f.second << std::endl;
	}

	std::ofstream o2(this->outputPath / fs::path("passtime.txt"));

	for (auto &f : ptSort) {
		o2 << f.second << std::endl;
	}

	std::ofstream o3(this->outputPath / fs::path("accelerate.txt"));

	for (auto &f : accSort) {
		o3 << f.second << std::endl;
	}
}

