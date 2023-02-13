#include <iostream>
#include "./include/EM_PORT_API.h"
#include "utils.h"

//  斐波序列计算方法
EM_PORT_API(int) fib(int n) {
    utils::localStaticVar();
    if (n <= 1) {
        return n;
    }
    return fib(n - 1) + fib(n - 2);
}

int main() {
    std::cout << "application.cpp runing" << std::endl;
    std::cin.get();
    utils::Vecv vecv;
    auto& vec = vecv.getVec();
    std::cout << vec.m_x << std::endl;
}