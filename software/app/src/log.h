#pragma once
namespace logger {
	struct LogInfo {
		char* name;
		char* message;
		unsigned int result;
	};
	void out(const char* message);
	void out(const char* meesage, const char* name);
	void out(const char* message, int value);
	void out(LogInfo logInfo);
	int multiply(int a, int b);
};