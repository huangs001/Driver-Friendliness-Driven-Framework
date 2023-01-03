#pragma once
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <filesystem>
#include <set>
#include <vector>
#include <tuple>
#include "DataLine.h"

class AdjGraphFilter
{
private:
	std::filesystem::path csvPath;
	std::filesystem::path osmidPath;
	std::filesystem::path nodeAdjPath;
	std::filesystem::path edgeAdjPath;
	std::unordered_map<DataLine::IdType, std::unordered_set<DataLine::IdType>> nodeAdj;
	std::unordered_map<DataLine::IdType, std::unordered_set<DataLine::IdType>> edgeAdj;
	std::unordered_set<DataLine::IdType> edgeSet;

	void loadFromTxt(const std::filesystem::path &path, std::unordered_map<DataLine::IdType, std::unordered_set<DataLine::IdType>> &result) const;

	void saveToTxt(const std::filesystem::path &path, const std::unordered_map<DataLine::IdType, std::unordered_set<DataLine::IdType>> &result) const;

	std::unordered_map<DataLine::IdType, std::set<std::tuple<DataLine::IdType, DataLine::IdType>>> filterEdge(const std::vector<std::vector<DataLine>> &trajectories, std::filesystem::path helper) const;

	std::unordered_map<DataLine::IdType, std::unordered_set<DataLine::IdType>> filterAdjGraph(const std::unordered_map<DataLine::IdType, std::set<std::tuple<DataLine::IdType, DataLine::IdType>>> &filterEdges) const;
public:
	AdjGraphFilter(std::filesystem::path csvPath, std::filesystem::path osmidPath, std::filesystem::path nodeAdjPath, std::filesystem::path edge2edgeAdjPath);

	std::unordered_set<DataLine::IdType> edgeOsmIdSet() const;

	void startFilter(unsigned threadNum, std::size_t limit, const std::vector<std::filesystem::path> &idPaths, const std::filesystem::path &outPath, const std::vector<std::filesystem::path> &outName = {}) const;
};

