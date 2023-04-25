@[TOC](Kotlin-极速基础入门)

Kotlin本质是适合带有一定Java基础或者偏现代语法例如Typescript会更容易上手。

几个特点先说一下：

* 多为Lambda表达式
* 防止null值出现
* 行末无须分号结束
* 多使用花括号{}作为载体体现语法特性
* 无自动隐形转换，例如整型1加字符串"1"在Java等语言中会自动转成字符串，但在Kotlin里会报错，要求必须手动将整型1转换成字符串才行

# 基础类型

同样包含Byte、Short、Int、Long、Float、Double，和Java不同的是，只有这种包装类，没有byte、int、short、float、double的原始类型。

```kotlin
//  常量
val con1 = 1
//  相当于
val con1: Int = 1
//  常量不可修改值
//con1 = 2
//  变量
var con2 = 1
//  同样相当于
var con2: Int = 1
//  变量可修改值
con2 = 2
//con2 = "1"  //  No allow, type are Int
//  字符串，可拼接变量
var cont3 = "this is string with value: ${con1}"
//  相当于
var cont3: String = "this is string with value: " + con1
//  多行字符串和Java一样使用```包裹即可
/*var cont4 = ```
    multiply
    line
    string
```*/
//  不可变数组
var arr = arrayOf(1, 2, 3)
//  可变列表
var arr = mutableListof(1, 2, 3)
```

# 判断

鼓励用when、is、else代替if、else和switch：
if判断：

```kotlin
var i = 2
var j = when (i) {
    is Int -> 20
    else -> "20"
}
```

switch判断：

```kotlin
var i = 2
var j = when (i) {
    1 -> 10
    2 -> 20
    3 -> 30
    else -> "as default"
}
```

# 循环

没有普通的for循环，只有范围for、增强for、数组类型的迭代器，普通for循环只能用while代替

## 范围for

```kotlin
//  范围循环：从某个数到另外一个数，默认步进值为1，每次循环自增1
for (i in 0 until 10) {
    println(i)  //  输出 0,1,2,3,4,5,6,7,8,9,10
}
//  范围循环：从某个数到另外一个数，并指定步进值，每次循环按步进值自增
for (i in 0 until 10 step 2) {
    println(i)  //  输出 0,2,4,6,8,10
}
//  范围循环，但是自减
for (i in 10 downTo 0) {
    println(i)  //  输出 10,9,8,7,6,5,4,3,2,1,0
}
//  范围循环，使用自动判断增减的省略号..
for (i in 0..10) {
    println(i)  //  输出 0,1,2,3,4,5,6,7,8,9,10
}
```

## 增强for

```kotlin
var arr = arrayOf(1, 2, 3)
for (value in arr) {
    println(value)
}
```

输出索引

```kotlin
var arr = arrayOf(1, 2, 3)
for (index in arr.indices) {
    println(index)
}
```

输出索引和值

```kotlin
var arr = arrayOf(1, 2, 3)
for (it in arr.valueWithIndex()) {
    println("${it.index}:${it.value}")
}
```

## 迭代器

输出值

```kotlin
var arr = arrayOf("hello", "world", "please")
//  直接输出值，第一个参数会自动命名为it传入
arr.forEach {
    println(it)
}
//  上面方式与以下相同
arr.forEach({ index -> {
        println(index)
}})
```

输出值和索引

```kotlin
//  输出索引和值，使用匿名方法
arr.forEachIndexed { index, value ->
    println(index, value)
}
//  输出索引和值，定义方法并引入使用
fun iteratorInt(index: Int, value: Int) {
    println(index, value)
}
arr.forEachIndexed(::iteratorInt)
```

## while

```kotlin
var i = 0
var length = 10
while (i < length) {
    println(i++)
}
```

## 跳出外层循环

采用对循环进行命名，之后指定名称进行break跳出。

```kotlin
var sum = 0
out@ for (i in range(1, 10)) {
    for (j in range(1, 4)) {
        sum *= (i + j)
        when {
            sum > 100 -> break@out
        }
    }
}
println("跳出后到了这里")
```

上面代码通过`名称@`的方式指定了循环体的名称，之后采用`break@名称`的方式指定要跳出的循环。

# 函数

相比Java而言重新体现函数的重要性，使用函数代替大部分Java里的类定义，其中就包括了main函数（Java里只能定义成类里的main方法）

```kotlin
fun handle(value: Int): Int {
    return value * value * 2
}
fun main() {
    println(handle(2))
}
```

# 类与接口

类无须new即可创建，注意内部依旧是使用new，与C++不一样，而是保持与Java等自动指针管理的开发语言一样。
在类名后添加括号可填写默认构造函数的形参列表，需要重构方法则在内部使用constructor关键字即可

采用冒号形式表示继承（类）和实现（接口），其中继承类需要加上小括号表示执行构造函数，并且要注意，默认所有类都是不可被继承的，所有方法都是不可被重写的，类和方法会自动添加final关键字，需要在被继承的类和被重写的方法前用open关键字修饰，而重写的新方法则用override关键字修饰

```kotlin
interface Live {
    fun info(): String
}
open class Person {
    private val name: String = "human"
}
class Man : Person(), Live {
    fun info(): String = "This is $name"
}
```

```kotlin
interface Live {
    fun info(): String
}
class Person(var name: String, var age: Int) : Live {
    private var name
        get() {
            return "The name is : $name"
        }
        set(value) {
            name = value
        }
    private var age

    constructor(name: String) : this(name, 1)
    constructor() : this("无名氏")

    fun info(): String {
        return this.toString()
    }
}

//  通过添加data修饰符可以让类自动重写toString, composeTo, hashCode等方法
//  通过添加by可以使用委托的形式让其他类完成所需的事情，例如此处让必须实现的方法info委托给Person处理，Man可以选择不实现
data class Man : Live by Person {
    fun work() {
        println("Go to work")
    }
}
```

另一种通过by方式实现属性值的委托，包含了[重载运算符](https://blog.csdn.net/qq_32677531/article/details/127336188)
特性重载了赋值和获取值的行为

```kotlin

class FamilyChoose {
    operator fun getValue(thisReference: Any, property: KProperty<*>): String {
        return v
    }
    operator fun setValue(thisReference: Any, property: KProperty<*>, i: String) {
        v = i
    }
}
class Woman : Person {
    var workType by FamilyChoose()
}
```

# 枚举类

通过枚举类完成罗列所需的有限集合。
属性值实际是继承自一个object类的实例，所以可以进行各种类操作，例如用于比较、输出字符串等。
由于属性值都是实例，所以在任何地方引入使用都是相同值，但实例也造成了必须长时间占用堆内存无法回收。

```kotlin
enum class Type {
    TEACHER,
    WORKER,
    ENGINEER,
    DESIGNER
}

val isSame = Type.TEACHER.equals(Type.WORKER)
```

# 密封类

属于枚举类的扩展类，相比起来属性值直接使用了类本身而不再是实例，当在使用时才创建实例，所以可被回收，虽然每次创建的实例不同，但可以直接对比类本身来确认是否相同

```kotlin
sealed class PartTimeJob {
    class DRIVER : Boolean
    class BATMAN : Boolean
}

fun handle(partTimeJob: PartTimeJob) {
    when (partTimeJob) {
        is PartTimeJob.BATMAN -> println("I'm batman")
        else -> println("Just a normal guy trying to live")
    }
}
fun main(wantToByHero: Boolean) {
    val partTimeJob = if (wantToByHero) PartTimeJob.BATMAN() else PartTimeJob.DRIVER()
    handle(partTimeJob)
}
```