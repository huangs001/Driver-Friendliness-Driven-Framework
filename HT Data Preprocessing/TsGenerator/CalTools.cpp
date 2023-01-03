#include "CalTools.h"
#include <cmath>
#include <ctime>

double CalTools::deg2rad(double deg)
{
    return deg * (CalTools::PI / 180);
}

double CalTools::getSpeedFromLatLonInMeterPerSecond(double lat1, double lon1, double lat2, double lon2, std::time_t tds)
{
    double dist = CalTools::getDistanceFromLatLonInKm(lat1, lon1, lat2, lon2);
    double speed = dist / tds * 1000;

    return speed;
}

double CalTools::getDistanceFromLatLonInKm(double lat1, double lon1, double lat2, double lon2)
{
    int R = 6371;
    double dLat = CalTools::deg2rad(lat2 - lat1);
    double dLon = CalTools::deg2rad(lon2 - lon1);
    double a = std::sin(dLat / 2) * std::sin(dLat / 2) + std::cos(CalTools::deg2rad(lat1)) * std::cos(CalTools::deg2rad(lat2)) * std::sin(dLon / 2) * std::sin(dLon / 2);
    double c = 2 * std::atan2(std::sqrt(a), std::sqrt(1 - a));
    double d = R * c;
    return d;
}
