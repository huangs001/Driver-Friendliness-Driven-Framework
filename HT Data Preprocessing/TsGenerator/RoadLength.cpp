#include "RoadLength.h"

#include <fstream>
#include <sstream>
#include <string>

RoadLength::RoadLength(const std::filesystem::path& osmRoadPath) {
	std::ifstream ifs(osmRoadPath);
	std::string line;

	while (std::getline(ifs, line)) {
		std::stringstream ss(line);
		std::string id, dist;
		std::getline(ss, id, ',');
		std::getline(ss, dist, ',');

		this->roadLength.emplace(std::stoul(id), std::stod(dist));
	}
}

double RoadLength::getRoadLength(unsigned long id) {
	return roadLength[id];
}
