#include "./include/EM_PORT_API.h"
#include "./include/stdafx.h"
#include "logger.h"
#include "utils.h"

//  쳲����м��㷽��
EM_PORT_API(int) fib(int n) {
    utils::localStaticVar();
    if (n <= 1) {
        return n;
    }
    return fib(n - 1) + fib(n - 2);
}

namespace allocateMetrics {
    static uint32_t s_allocCount = 0;
}

//  ��дnew�����������ж��ڴ���䶼�ܽ��м�¼���Ա���г����Ż�
//void* operator new (size_t size) {
//    allocateMetrics::s_allocCount++;
//    std::cout << "allocating " << size << " bytes\n";
//    return malloc(size);
//}

//void operator delete(void* memory, size_t size) {
//    allocateMetrics::s_allocCount--;
//    std::cout << "Freeing " << size << " bytes\n";
//    free(memory);
//}

int main() {
    logger::out("application.cpp runing");
    utils::initListNumberAdd();
}