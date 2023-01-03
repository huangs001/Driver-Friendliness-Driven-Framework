#include "AdjGraphFilter.h"
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <tuple>
#include "DataLine.h"
#include "DataParser.h"
#include "SliceMultiThread.h"

void AdjGraphFilter::loadFromTxt(const std::filesystem::path &path, std::unordered_map<DataLine::IdType, std::unordered_set<DataLine::IdType>> &result) const
{
	std::ifstream if1(path);
	std::string line;
	std::string subline;
	while (std::getline(if1, line)) {
		auto mainId = DataLine::castid(line);
		std::getline(if1, subline);
		std::stringstream ss1(subline);
		DataLine::IdType subid = 0;
		auto &oneAdj = result[mainId];
		while (ss1 >> subid) {
			oneAdj.insert(subid);
		}
	}
}

void AdjGraphFilter::saveToTxt(const std::filesystem::path &path,
	const std::unordered_map<DataLine::IdType, std::unordered_set<DataLine::IdType>> &result) const {

	std::ofstream of1(path);
	std::string sep = " ";

	for (auto &node : result) {
		of1 << node.first << std::endl;
		auto &adj = node.second;
		auto begin = adj.begin(), end = adj.end();
		if (!adj.empty()) {
			of1 << *begin++;
		}
		while (begin != end) {
			of1 << sep << *begin++;
		}
		of1 << std::endl;
	}
}

std::unordered_map<DataLine::IdType, std::set<std::tuple<DataLine::IdType, DataLine::IdType>>> AdjGraphFilter::filterEdge(const std::vector<std::vector<DataLine>> &trajectories, std::filesystem::path helper) const
{
	std::unordered_map<DataLine::IdType, std::set<std::tuple<DataLine::IdType, DataLine::IdType>>> results;

	std::unordered_map<DataLine::IdType, unsigned> passCnt;

	for (auto &trj : trajectories) {
		for (auto &line : trj) {
			++passCnt[line.osmid[0]];
		}
	}

	std::ofstream ofs(helper);
	for (auto& kv : passCnt) {
		ofs << kv.first << " " << kv.second << std::endl;
	}

	for (auto &node : this->nodeAdj) {
		auto &adj = node.second;
		if (adj.size() >= 3) {
			std::vector<std::tuple<unsigned, DataLine::IdType>> adjCnt;

			unsigned zeroCnt = 0;
			for (auto id : adj) {
				auto cnt = passCnt[id];
				adjCnt.emplace_back(cnt, id);
				if (cnt == 0) {
					++zeroCnt;
				}
			}
			//if (zeroCnt >= adj.size() - 3) {
			//	continue;
			//}
			
			std::vector<std::tuple<unsigned, DataLine::IdType, DataLine::IdType>> helper;
			for (std::size_t i = 0; i < adjCnt.size(); ++i) {
				for (std::size_t j = i + 1; j < adjCnt.size(); ++j) {
					helper.emplace_back(std::get<0>(adjCnt[i]) + std::get<0>(adjCnt[j]), std::get<1>(adjCnt[i]), std::get<1>(adjCnt[j]));
				}
			}

			std::sort(helper.begin(), helper.end());

			unsigned sum = 0, sumsum = 0;
			for (auto t : helper) {
				auto num = std::get<0>(t);
				sum += num;
				sumsum += num * num;
			}

			double avg = 1.0 * sum / static_cast<double>(adjCnt.size());
			double dev = std::sqrt((sumsum - (sum * sum / static_cast<double>(adjCnt.size()))) / static_cast<double>(adjCnt.size()));

			auto clip = adj.size() - 2;
			for (decltype(clip) i = 0; i < clip; ++i) {
				if (std::get<0>(helper[i]) <= avg - dev) {
					auto &first = std::get<1>(helper[i]);
					auto &second = std::get<2>(helper[i]);
					results[node.first].insert(first < second ? std::make_tuple(first, second) : std::make_tuple(second, first));
				}
			}

			
			//double avg = 1.0 * sum / static_cast<double>(adjCnt.size());
			//double dev = (sumsum - (sum * sum / static_cast<double>(adjCnt.size()))) / static_cast<double>(adjCnt.size());
			//double limit = avg - 2 * dev;
			//for (std::size_t i = 0; i < adjCnt.size(); ++i) {
			//	if (adjCnt[i] < limit) {
			//		results.emplace_back(node.first, idList[i]);
			//	}
			//}
		}
	}

	return results;
}

std::unordered_map<DataLine::IdType, std::unordered_set<DataLine::IdType>> AdjGraphFilter::filterAdjGraph(
	const std::unordered_map<DataLine::IdType, std::set<std::tuple<DataLine::IdType, DataLine::IdType>>> &filterEdges) const {

	std::unordered_map<DataLine::IdType, std::unordered_set<DataLine::IdType>> newGraph;

	for (auto &edge : edgeAdj) {
		auto &adj = newGraph[edge.first];
		
		for (auto &subNode : edge.second) {
			auto checkSetIt = filterEdges.find(subNode);
			
			for (auto &subEdge : nodeAdj.at(subNode)) {
				auto &first = edge.first;
				auto &second = subEdge;
				if (subEdge == edge.first || (checkSetIt != filterEdges.end() && checkSetIt->second.count(first < second ? std::make_tuple(first, second) : std::make_tuple(second, first)) > 0)) {
					if (subEdge != edge.first)
						std::cout << "We filter " << edge.first << "'s " << subEdge << std::endl;
					continue;
				}
				adj.emplace(subEdge);
			}
		}
	}

	return newGraph;
}

AdjGraphFilter::AdjGraphFilter(std::filesystem::path csvPath, std::filesystem::path osmidPath, std::filesystem::path nodeAdjPath, std::filesystem::path edgeAdjPath)
	:csvPath(std::move(csvPath)), osmidPath(std::move(osmidPath)), nodeAdjPath(nodeAdjPath), edgeAdjPath(edgeAdjPath)
{
	loadFromTxt(nodeAdjPath, nodeAdj);
	loadFromTxt(edgeAdjPath, edgeAdj);
	for (auto &edge : edgeAdj) {
		edgeSet.insert(edge.first);
	}
}


std::unordered_set<DataLine::IdType> AdjGraphFilter::edgeOsmIdSet() const {
	return edgeSet;
}

void AdjGraphFilter::startFilter(unsigned threadNum, std::size_t limit,
	const std::vector<std::filesystem::path> &idPaths, const std::filesystem::path &outPath,
	const std::vector<std::filesystem::path> &outName) const {

	DataParser dataParser(this->csvPath, this->osmidPath);

	SliceMultiThread::multiThread(idPaths, threadNum, [&](unsigned, std::size_t idx, const std::filesystem::path &path) {
		std::vector<std::vector<DataLine>> trjs;
		std::ifstream if1(path);
		std::size_t id = 0;

		std::size_t limitCnt = 0;
		while (if1 >> id) {
			auto trj = dataParser.parse(id);
			trjs.emplace_back(trj);
			if (++limitCnt >= limit) {
				break;
			}
		}
		std::cout << "go" << std::endl;
		saveToTxt(outPath / (idx < outName.size() ? outName[idx] : std::filesystem::path(std::to_string(idx) + ".txt")), filterAdjGraph(filterEdge(trjs, outPath / std::filesystem::path(std::to_string(idx) + "_cnt.txt"))));
	});
}
