#include "./include/stdafx.h"

namespace allocate_metrics {
	static uint32_t s_allocCount = 0;
}

//  ��дnew�����������ж��ڴ���䶼�ܽ��м�¼���Ա���г����Ż�
//void* operator new (size_t size) {
//    allocate_metrics::s_allocCount++;
//    std::cout << "allocating " << size << " bytes\n";
//    return malloc(size);
//}

//void operator delete(void* memory, size_t size) {
//    allocate_metrics::s_allocCount--;
//    std::cout << "Freeing " << size << " bytes\n";
//    free(memory);
//}