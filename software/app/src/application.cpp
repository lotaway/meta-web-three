#include <iostream>
#include "./include/EM_PORT_API.h"
#include "logger.h"
#include "utils.h"

// 通过条件判断决定最后调用LOG的代码会被替换成命令行输出或者空白，而判断的条件同样通过修改宏的值决定，也可以在项目属性》C / C++》预处理器》预处理器定义里，通过设置Debug和Release不同配置，并添加MODE＝1;来指定Debug配置下使用的值。这样调试Debug模式时就能输出，而调试Release或者发布程序时就不用自行修改代码里的MODE值。
//#define MODE 1
#if MODE==1
#define LOG(str) std::cout << str << std::endl;
#else
#define LOG(str)
#endif

//  斐波序列计算方法
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