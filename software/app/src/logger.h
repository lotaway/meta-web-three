#pragma once
namespace logger {

#define PROFILING 1
#if PROFILING
#define PROFILE_SCOPE(name) logger::timer timer##__LINE__(name);
#define PROFILE_FUNCTION() PROFILE_SCOPE(__FUNCTION__)
#else
#define PROFILE_SCOPE(name)
#define PROFILE_FUNCTION()
#endif

	//	��׼���ԣ�ͨ�����ü�ʱ��/���������������/�����ռ���¼����/���������������ڹ��캯��������������ɳ���ʱ�����
	struct timer {
		const char* name;
		std::chrono::high_resolution_clock::time_point start, end;
		std::chrono::duration<float> duration;
		timer(const char* _name);
		~timer();
	};
	struct LogInfo {
		char* name;
		char* message;
		unsigned int result;
	};
	class Info {
		void test();
	};
	void out(const int value);
	void out(const char* message);
	void out(const char* meesage, const char* name);
	void out(const char* message, int value);
	void out(const std::string& message);
	void out(const std::string& message, int value);
	void out(LogInfo logInfo);
};