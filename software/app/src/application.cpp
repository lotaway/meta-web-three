#include <iostream>
#include "./include/EM_PORT_API.h"
#include "logger.h"
#include "utils.h"

// ͨ�������жϾ���������LOG�Ĵ���ᱻ�滻��������������߿հף����жϵ�����ͬ��ͨ���޸ĺ��ֵ������Ҳ��������Ŀ���ԡ�C / C++��Ԥ��������Ԥ�����������ͨ������Debug��Release��ͬ���ã������MODE��1;��ָ��Debug������ʹ�õ�ֵ����������Debugģʽʱ���������������Release���߷�������ʱ�Ͳ��������޸Ĵ������MODEֵ��
//#define MODE 1
#if MODE==1
#define LOG(str) std::cout << str << std::endl;
#else
#define LOG(str)
#endif

//  쳲����м��㷽��
EM_PORT_API(int) fib(int n) {
    utils::localStaticVar();
    if (n <= 1) {
        return n;
    }
    return fib(n - 1) + fib(n - 2);
}

int main() {
    LOG("application.cpp runing");
    std::cin.get();
}