#include <iostream>
#include "./include/EM_PORT_API.h"
#include "log.h"
using namespace std;

int g_variable = 5;	//	全局变量，将可以被extern装饰的同名变量所引用

namespace logger {
	void out(const char* message) {
		cout << message << endl;
	}

	void out(const char* message, const char* name) {
		out(name);
		out(message);
	}

	void out(const char* message, const int value) {
		out(message);
		cout << value << endl;
	}

	void out(LogInfo logInfo) {
		out(logInfo.message);
		out(logInfo.name);
	}

	int multiply(int a, int b) {
		return a * b;
	}
}