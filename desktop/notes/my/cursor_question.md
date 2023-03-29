# how to transfer vector<char> to string in c++ ?

To transfer a vector<char> to a string in C++, you can use the constructor of the string class that takes two iterators as arguments. The first iterator should point to the beginning of the vector<char>, and the second iterator should point to the end of the vector<char>. Here's an example:

```C++
#include <iostream>
#include <string>
#include <vector>

int main() {
    std::vector<char> vec = {'h', 'e', 'l', 'l', 'o'};
    std::string str(vec.begin(), vec.end());
    std::cout << str << std::endl;
    return 0;
}
```
In this example, we create a vector<char> called vec that contains the characters 'h', 'e', 'l', 'l', and 'o'. We then create a string called str using the constructor that takes two iterators. The first iterator is vec.begin(), which points to the beginning of the vector<char>, and the second iterator is vec.end(), which points to one past the end of the vector<char>. This creates a string that contains the same characters as the vector<char>.

When we run this program, it outputs:

hello

I hope this helps! Let me know if you have any other questions.

# why double look different to int type in C++ memory ?
Double and int types are stored differently in C++ memory because they have different sizes and representations. Integers are typically stored as 32-bit values, while doubles are stored as 64-bit values. Additionally, the representation of a double is different from that of an int. Integers are stored as binary values, while doubles are stored using a floating-point representation. This means that the bits that make up a double value are divided into three parts: the sign bit, the exponent, and the mantissa. The sign bit indicates whether the value is positive or negative, the exponent determines the scale of the value, and the mantissa contains the actual digits of the value. Because of these differences in size and representation, doubles and ints look different in C++ memory.

# 为什么double和int类型在C++内存里看起来不一样？用中文回答
Double和int类型在C++内存里看起来不一样是因为它们的大小和表示方式不同。整数通常被存储为32位值，而双精度浮点数被存储为64位值。此外，双精度浮点数的表示方式与整数的表示方式不同。整数以二进制值存储，而双精度浮点数使用浮点表示法存储。这意味着组成双精度浮点数值的位被分为三个部分：符号位、指数和尾数。符号位表示值是正数还是负数，指数确定值的比例，尾数包含值的实际数字。由于这些大小和表示方式的差异，双精度浮点数和整数在C++内存中看起来不同。