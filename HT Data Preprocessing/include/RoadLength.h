#pragma once
#include <filesystem>
#include <unordered_map>

class RoadLength
{
private:
	std::unordered_map<unsigned long, double> roadLength;
public:
	RoadLength(const std::filesystem::path& osmRoadPath);
	double getRoadLength(unsigned long id);
};

