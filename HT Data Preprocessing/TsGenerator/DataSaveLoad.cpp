#include "DataSaveLoad.h"
#include <fstream>

void DataSaveLoad::dumpData(const TSDataType& osmData, const std::string& path)
{
	dumpData(osmData, path, [](IdType, const std::vector<InnerType> &) { return true; });
}

DataSaveLoad::TSDataType DataSaveLoad::loadData(const std::string& path)
{
	TSDataType osmData;
	std::ifstream ifs(path, std::ios::binary);

	unsigned osmid = 0;
	while (ifs.read(reinterpret_cast<char*>(&osmid), sizeof(IdType))) {
		std::size_t length = 0;
		ifs.read(reinterpret_cast<char*>(&length), sizeof(std::size_t));
		std::vector<InnerType> datas;
		for (std::size_t i = 0; i < length; ++i) {
			CountType data1 = 0;
			ValueType data2 = 0;
			ifs.read(reinterpret_cast<char*>(&data1), sizeof(CountType));
			ifs.read(reinterpret_cast<char*>(&data2), sizeof(ValueType));
			datas.emplace_back(data1, data2);
		}
		osmData.emplace(osmid, datas);
	}

	return osmData;
}
