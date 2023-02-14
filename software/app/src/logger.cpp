#include <iostream>
#include "logger.h"
using namespace std;

int g_variable = 5;	//	ȫ�ֱ����������Ա�externװ�ε�ͬ������������

namespace logger {

	void Info::test() {

	}

	void out(const int value) {
		cout << value << endl;
	}

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