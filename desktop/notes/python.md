# Python基本语法介绍

## 变量

Python中的变量不需要声明，直接赋值即可。变量的类型会根据赋值自动确定。

```python
a = 1
b = "hello"
c = True
```

## 数据类型

Python中常见的数据类型有整型、浮点型、字符串、布尔型、列表、元组、字典等。

```python
a = 1  # 整型
b = 1.0  # 浮点型
c = "hello"  # 字符串
d = True  # 布尔型
e = [1, 2, 3]  # 列表
f = (1, 2, 3)  # 元组
g = {"name": "Tom", "age": 18}  # 字典
```

## 运算符

Python中常见的运算符有算术运算符、比较运算符、逻辑运算符等。

```python
a = 1 + 2  # 加法运算
b = 3 - 2  # 减法运算
c = 2 * 3  # 乘法运算
d = 4 / 2  # 除法运算
e = 5 % 2  # 取模运算
f = 2 ** 3  # 幂运算
g = 3 > 2  # 大于运算
h = 3 == 2  # 等于运算
i = not True  # 非运算
j = True and False  # 与运算
k = True or False  # 或运算
```

## 控制语句

Python中常见的控制语句有if语句、for循环、while循环等。

```python
# if语句
if a > b:
    print("a > b")
elif a == b:
    print("a == b")
else:
    print("a < b")

# for循环
for i in range(10):
    print(i)

# while
while condition:
    # do something
# 函数

Python中可以使用def关键字定义函数，函数可以接受参数，也可以返回值。

```python
def add(a, b):
    return a + b

result = add(1, 2)
print(result)
```

# 类

Python中可以使用class关键字定义类，类可以包含属性和方法。

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def say_hello(self):
        print("Hello, my name is", self.name)

p = Person("Tom", 18)
p.say_hello()
```

# 异常处理

Python中可以使用try...except...finally语句来处理异常。

```python
try:
    # do something
except Exception as e:
    # handle exception
finally:
    # do something finally
```

# 文件操作

Python中可以使用open函数来打开文件，使用with语句来自动关闭文件。

```python
with open("file.txt", "r") as f:
    content = f.read()
    print(content)
```

# 模块

Python中可以使用import语句来导入模块，也可以使用from...import语句来导入模块中的指定内容。

```python
import math

result = math.sqrt(2)
print(result)

from datetime import datetime

now = datetime.now()
print(now)
