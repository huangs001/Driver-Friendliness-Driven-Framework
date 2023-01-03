#ifndef FRIEND_WEIGHT_H
#define FRIEND_WEIGHT_H

namespace RoutingKit {
struct friend_weight {
    unsigned travel_time;
    unsigned safety;
    unsigned comfort;
    unsigned length;

    bool operator==(const friend_weight &rhs) const {
        return travel_time == rhs.travel_time && (safety + comfort == rhs.safety + rhs.comfort);
    }

    bool operator!=(const friend_weight &rhs) const {
        return !(*this == rhs);
    }

    bool operator<(const friend_weight &rhs) const {
        return travel_time != rhs.travel_time ? travel_time < rhs.travel_time : safety + comfort < rhs.safety + rhs.comfort;
    }

    bool isInf() const {
        return travel_time == 2147483647u;
    }

    bool operator<=(const friend_weight &rhs) const {
        return *this == rhs || *this < rhs;
    }

    bool operator>(const friend_weight &rhs) const {
        return !(*this <= rhs);
    }

    bool operator>=(const friend_weight &rhs) const {
        return !(*this < rhs);
    }

    friend_weight():travel_time(0), safety(0), comfort(0), length(0) {
    }

    friend_weight(unsigned travel_time, unsigned safety, unsigned comfort, unsigned length):
        travel_time(travel_time), safety(safety), comfort(comfort), length(length) {
    }

    friend_weight(unsigned travel_time):
        travel_time(travel_time), safety(travel_time), comfort(travel_time), length(travel_time) {
    }

    friend_weight& operator=(const friend_weight &rhs) {
        this->travel_time = rhs.travel_time;
        this->safety = rhs.safety;
        this->comfort = rhs.comfort;
        this->length = rhs.length;
        return *this;
    }

    friend_weight& operator=(const unsigned val) {
        this->travel_time = val;
        this->safety = val;
        this->comfort = val;
        this->length = val;
        return *this;
    }

    friend_weight operator+(const friend_weight &rhs) const {
        return friend_weight{travel_time + rhs.travel_time, safety + rhs.safety, comfort + rhs.comfort, length + rhs.length};
    }
};
}

#endif
