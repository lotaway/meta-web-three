#pragma once
struct LogInfo {
	char* name;
	char* message;
	unsigned int result;
};
void log(const char* message);
void log(const char* meesage, const char* name);
void log(const char* message, int value);
void log(LogInfo logInfo);

int multiply(int a, int b);