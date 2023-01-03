#pragma once
#include <cmath>
#include <ctime>

class CalTools
{
private:
	static constexpr double PI = 3.14159265358979323846;
	static double deg2rad(double deg);
public:
	static double getSpeedFromLatLonInMeterPerSecond(double lat1, double lon1, double lat2, double lon2, std::time_t tds);
	static double getDistanceFromLatLonInKm(double lat1, double lon1, double lat2, double lon2);
};

