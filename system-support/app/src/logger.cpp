#include <iostream>
#include "./include/stdafx.h"
#include "logger.h"
using namespace std;

int g_variable = 5;	//	ȫ�ֱ����������Ա�externװ�ε�ͬ������������

namespace logger {
	
	timer::timer(const char* _name) : name(_name) {
		start = end = std::chrono::high_resolution_clock::now();
		duration = end - start;
	};
	
	timer::~timer() {
		end = std::chrono::high_resolution_clock::now();
		duration = end - start;
		//	ʹ��\nҲ���<< std::endl���죨��5-8ms��
		std::cout << name + ':' <<  + duration.count() * 1000.0f << std::endl;
		//	����ʱ�Զ������ϵ�
		__debugbreak();
	};

	void info::test() {

	}

	void out(const int value) {
		cout << value << endl;
	}

	void into_file(const char* message) {
		//	record into file
	}

	// ͨ�������жϾ���������LOG�Ĵ���ᱻ�滻��������������߿հף����жϵ�����ͬ��ͨ���޸ĺ��ֵ������Ҳ��������Ŀ���ԡ�C / C++��Ԥ��������Ԥ�����������ͨ������Debug��Release��ͬ���ã������MODE��1;��ָ��Debug������ʹ�õ�ֵ����������Debugģʽʱ���������������Release���߷�������ʱ�Ͳ��������޸Ĵ������MODEֵ��
	//#define MODE 1
	#if MODE==1
	#define modeOut(message) cout << message << endl;
	#else
	#define modeOut(message) intoFile(message)
	#endif

	void out(const char* message) {
		modeOut(message);
	}

	void out(const char* message, const char* name) {
		out(name);
		out(message);
	}

	void out(const char* message, const int value) {
		out(message);
		out(value);
	}

	void out(const std::string& message) {
		modeOut(message);
	}

	void out(const std::string& message, const int value) {
		out(message);
		out(value);
	}

	void out(log_info l_info) {
		out(l_info.message);
		out(l_info.name);
	}
}