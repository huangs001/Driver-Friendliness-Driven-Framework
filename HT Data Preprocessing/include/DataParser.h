#pragma once
#include <string>
#include <functional>
#include <filesystem>
#include <vector>
#include "DataLine.h"

class DataParser
{
private:
	std::filesystem::path csvFolder;
	std::filesystem::path osmidFolder;
	std::vector<std::filesystem::path> filePaths;
public:
	DataParser(std::filesystem::path csvFolder, std::filesystem::path osmidFolder);

	void parse(std::function<void(std::size_t, std::size_t, const std::vector<DataLine> &)> fun, unsigned threadNum);

	std::size_t size();

	std::vector<DataLine> parse(std::size_t fileNum);
};

