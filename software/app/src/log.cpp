#include <iostream>
#include "./include/EM_PORT_API.h"
#include "log.h"
using namespace std;

int g_variable = 5;	//	ȫ�ֱ����������Ա�externװ�ε�ͬ������������

void log(const char* message) {
	cout << message << endl;
}

void log(const char* message, const char* name) {
	log(name);
	log(message);
}

void log(const char* message, const int value) {
	log(message);
	cout << value << endl;
}

void log(LogInfo logInfo) {
	log(logInfo.message);
	log(logInfo.name);
}

int multiply(int a, int b) {
	return a * b;
}