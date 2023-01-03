#pragma once
#include <unordered_map>
#include <fstream>
#include <vector>

class DataSaveLoad
{
public:
	typedef int CountType;
	typedef double ValueType;
	typedef unsigned IdType;
	typedef std::pair<CountType, ValueType> InnerType;
	typedef std::unordered_map<IdType, std::vector<InnerType>> TSDataType;

	static void dumpData(const TSDataType& osmData, const std::string& path);

	static TSDataType loadData(const std::string& path);

	template <typename Fun>
	static void dumpData(const TSDataType &osmData, const std::string &path, Fun checkFun);

	template <typename Fun>
	static void dumpPlainData(const TSDataType &osmData, const std::string &path, Fun mergeFun);
};

template <typename Fun>
void DataSaveLoad::dumpData(const TSDataType &osmData, const std::string &path, Fun checkFun) {
	std::ofstream partOut(path, std::ios::binary);
	for (const auto &kv : osmData) {
		if (!checkFun(kv.first, kv.second)) {
			continue;
		}
		partOut.write(reinterpret_cast<const char *>(&kv.first), sizeof(IdType));
		std::size_t length = kv.second.size();
		partOut.write(reinterpret_cast<const char *>(&length), sizeof(std::size_t));

		for (std::size_t i = 0; i < length; ++i) {
			CountType data1 = kv.second[i].first;
			ValueType data2 = kv.second[i].second;
			partOut.write(reinterpret_cast<char *>(&data1), sizeof(CountType));
			partOut.write(reinterpret_cast<char *>(&data2), sizeof(ValueType));
		}
	}
}

template <typename Fun>
void DataSaveLoad::dumpPlainData(const TSDataType &osmData, const std::string &path, Fun mergeFun) {
	std::ofstream partOut(path);
	for (const auto &kv : osmData) {
		partOut << kv.first << std::endl;

		if (!kv.second.empty()) {
			partOut << mergeFun(kv.first, kv.second[0].first, kv.second[0].second);
		}
		for (std::size_t i = 1; i < kv.second.size(); ++i) {
			partOut << " " << mergeFun(kv.first, kv.second[i].first, kv.second[i].second);
		}
		partOut << std::endl;
	}
}

