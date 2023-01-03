#pragma once
#include <vector>
#include <ctime>
#include <string>
class DataLine
{
public:
	typedef unsigned long long IdType;
	inline static IdType castid(const std::string &st) {
		return std::stoull(st);
	}

	IdType taxiID;
	std::time_t ts;
	double lon;
	double lat;
	std::vector<IdType> osmid;

	DataLine(): DataLine(0, 0, 0, 0, std::vector<IdType>()) {}
	DataLine(IdType taxiID, std::time_t ts, double lon, double lat, const std::vector<IdType> &osmid)
		:taxiID(taxiID),
		ts(ts),
		lon(lon),
		lat(lat),
		osmid(osmid) {}
};

