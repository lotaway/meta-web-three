#include "./include/EM_PORT_API.h"
#include "./include/stdafx.h"
#include "logger.h"
#include "utils.h"

//  斐波序列计算方法
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

//  重写new操作符让所有堆内存分配都能进行记录，以便进行程序优化
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