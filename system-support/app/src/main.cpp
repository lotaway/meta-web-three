#include "./include/EM_PORT_API.h"
#include "./include/stdafx.h"
#include "logger.h"
#include "utils.h"
#include "hazel.h"
extern "C"
{
#include "hello.h"
}

int main() {
    utils::test_atoi();
    test_hello();
}
