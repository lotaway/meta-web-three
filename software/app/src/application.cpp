#include <iostream>
#include "./include/EM_PORT_API.h"
#include "./utils.h"

//  쳲����м��㷽��
EM_PORT_API(int) fib(int n) {
    localStaticVar();
    if (n <= 1) {
        return n;
    }
    return fib(n - 1) + fib(n - 2);
}

int main() {
    std::cout << "application.cpp runing" << std::endl;
    std::cin.get();
}