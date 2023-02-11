#include <iostream>
#include "./utils.h"
#include "./log.h"
//	std:array 需要
#include <array>
//	cout << std::to_string 时需要，帮助int转为string
#include <string>
//	unique_ptr智能指针所需要引入
#include <memory>
//	std::vector类需要引入
#include <vector>
//	静态库，需要打包到exe文件中，效率更好，但单文件更大
//	动态库，一般是放置到exe文件旁边
//	引入依赖库
#include <GLFW/glfw3.h>
//	引入解决方案中的其他项目
// emsdk无法识别，只能使用引号加相对路径"../../engine/src/engine.h"，除了cpp和标准库以外的文件都没有被编译进去wasm
#include <engine.h>
//	make_tuple需要
#include <tuple>

#define debugger(msg) std::cout << "main::debugger:" + msg << std::endl

extern int g_variable;	//	将从外部所有文件里找到这个全局值，目前定义在log.cpp里

//	来源于静态库或动态库的方法
void useLibrary() {
	int a = glfwInit();
	std::cout << a << std::endl;
	engine::printMessage();
}

void variableAndLog() {
	log("hello world");
	const char* name = "c++";
	std::string s = "This's string~~";
	//std::wstring s = "这就是字符串了";
	log("welcome", name);
	int value = -1;
	unsigned int uValue = 2;
	float fVal = 5.5f;
	double dVal = 5.5;
	short sVal = 1;
	long lVal = 2;
	long int liVal = 3;
	bool needMoreInfo = value < 0;
	//log("going end", value);
	int mVal = multiply(2, 3);
	log("mVal", mVal);
	std::cin.get();

	if (needMoreInfo) {
		log("need more info");
		for (int i = 0;i < 10;i++) {
			value--;
			log("value", value);
		}
	}
	std::cin.get();

	int var = 8;
	void* ptr = &var;
	//*ptr = 10;	//	错误，void指针无法明白10是什么类型则不能写入
	int* ptr1 = &var;
	*ptr1 = 10;	//	正确，int指针明白10应当是整型类型，可以写入
	double* ptr2 = (double*)&var;
	//log((char*)&ptr2);
	std::cin.get();

	char* buffer = new char[8];	//	请求8字节内存
	memset(buffer, 0, 8);	//	用数据填充内存
	//	执行一些操作
	char** ptr3 = &buffer;
	//	执行一些操作之后。。。
	delete[] buffer;	//	删除指针，因为填充了内存
	std::cin.get();
}

//	强大：传递指针进行修改
void incrementWithPointer(int* value) {
	(*value)++;
}

//	简化：传递引用进行修改，推荐使用这种方式
void incrementWithReference(int& value) {
	value++;
}

void PointerAndReference() {
	int a = 5;
	int* b = &a;
	int c = 8;
	*b = 6;
	incrementWithPointer(b);
	log("incrementWithPointer b", *b);
	log("what about a", a);
	incrementWithPointer(&a);
	log("incrementWithPointer a", a);
	log("what about b", *b);
	incrementWithReference(a);
	log("incrementWithReference a", a);
	std::cin.get();

	int& ref = a;
	//	ref是对a的引用，将c这个值赋值给ref和a，即a=8，ref=8，并且c=8，两者互不干扰
	ref = c;
	log("ref", ref);
	//	指针可以被修改指向到新的地址，但引用不可以，只能赋值
	b = &c;
	c = 10;
	log("b change", *b);
}

void localStaticVar() {
	//	局部静态变量实际上是持续存在的静态变量，第一次调用会初始化（值为1），之后保持增长（2、3、4...）
	static int variable = 1;
	variable++;
	std::cout << "local static variable " + std::to_string(variable) << std::endl;
}

void initStatic() {
	log("extern global variable", g_variable);
	//	调用多次函数内将输出不同的局部静态变量
	localStaticVar();	//	2
	localStaticVar();	//	3
	localStaticVar();	//	4
}

enum PlayerLevel {
	PlayerLevel_EntryLevel,
	PlayerLevel_Intermediate,
	PlayerLevel_Expert
};

enum PlayerStatus {
	PlayerStatus_Enabled,
	PlayerStatus_Disabled
};

class Player {
private:
	PlayerLevel m_level;
	PlayerStatus m_status;
public:
	int m_positionX, m_positionY;
	int m_speed;
	//	构造函数，实例化时调用的方法，名称和类名一样，需要初始化所有的实例变量
	//	explicit关键字禁止隐性转换，如Player player = PlayerLevel_EntryLevel;
	explicit Player(PlayerLevel level = PlayerLevel_EntryLevel) : m_level(level), m_status(PlayerStatus_Enabled), m_positionX(0), m_positionY(0), m_speed(15) {

	}
	//	摧毁类实例时调用的方法，名称为【~类名】
	~Player() {
		std::cout << "go out" << std::endl;
	}
	void move(int new_x, int new_y) {
		m_positionX += new_x * m_speed;
		m_positionY += new_y * m_speed;
	}
};

void fastMove(Player& player, int new_x, int new_y) {
	player.m_positionX += new_x * player.m_speed * 2;
	player.m_positionY += new_y * player.m_speed * 2;
}

//	struct 是为了兼容c语法，与class的区别只有struct内的值默认是public，而class默认都是private
struct NormalPerson {
	int m_positionX, m_positionY;
	int m_speed;
	Player* m_like;
	void move(int new_x, int new_y) {
		m_positionX += new_x * m_speed;
		m_positionY += new_y * m_speed;
	}
	void follow(Player& _player) {
		m_like = &_player;
	}
};

//	访问修饰符
class Trainer {
	//	只能此类调用
private:
	int m_runLevel;
	int m_runNumber;
	//	可被此类和继承类调用
protected:
	int m_age;
	int m_sex;
	//	所有代码都可调用
public:
	Trainer(int runNumber, int age, int sex) {
		m_runLevel = 0;
		m_runNumber = runNumber;
		m_age = age;
		m_sex = sex;
	}
};
//	通过虚函数实现抽象类/接口
class Runner {
public:
	virtual void run() = 0;
};

//	继承
class Racer : public Runner {
public:
	char m_cup;
	int m_rank;
	//	初始化列表形式的构造函数
	Racer(const char& cup, int rank) : m_cup(cup), m_rank(rank) {}
	void run() override {
		std::cout << std::string("run") << std::endl;
	}
};

class Winner : public Racer {
public:
	std::string getNews() {
		return "He win.";
	}
};

class InitStatic {
public:
	static const int s_defaultSpeed = 2;
	static int s_maxSpeed;
};

//	初始化静态变量只能在类外的全局方式完成
//	方式1：通过类名设置
int InitStatic::s_maxSpeed = 3;
//	方式2：类实例里也能像设置成员变量一样设置静态变量
//InitStatic initState;
//int initState.s_maxSpeed = 3

void initClass() {
	//	在栈上实例化，只在作用域内存在，离开作用域后自动释放内存，且栈只有1~2MB大小
	Player* p1;
	{
		Player player;
		player.m_positionX = 0;
		player.m_positionY = 0;
		player.m_speed = 10;
		player.move(1, 1);
		fastMove(player, 1, 1);
		p1 = &player;
	}
	//	p1指向为空的Player类，因为原本的player已经销毁
	std::cout << p1 << std::endl;
	//	在堆上实例化，可以在作用域以外的地方存在，不使用时需要手动释放内存
	Player* p2;
	{
		Player* player2 = new Player();
		player2->m_positionX = 1;
		p2 = player2;
	}
	//	*p2指向的player2依旧存在
	std::cout << p2 << std::endl;

	// 通过对象方式直接初始化成员变量，只适用于没有自定义构造函数的情况
	NormalPerson normalPerson = { 0, 0, 10 };

	//	new符号初始化时只会返回指针，需要用*存储
	Player* player3 = new Player();
	player3->m_positionX = 0;

	NormalPerson* normalPerson2 = new NormalPerson();
	//	由于是指针，可以通过使用星号调用或设置成员变量
	(*normalPerson2).m_speed = 4;
	//	也可以通过箭头->调用或设置成员变量
	normalPerson2->m_speed = 5;
	//	如果不是独立的函数等作用域，例如main中分配了内存空间，不使用了要进行删除
	delete normalPerson2;

	std::cin.get();

}

void initArray() {
	const int arrSize = 5;
	//	定义定长整型数组，必须再通过循环进行初始化
	int arr[arrSize];
	std::cout << arr << std::endl;
	//	实例化定义，实例必须是指针，必须通过循环进行初始化
	int* iarr = new int[arrSize];
	//	不建议使用：获取整型数组长度（没有可以直接获取长度的属性或方法，sizeof获取的是字节长度）
	//int arrSize = sizeof(iarr) / sizeof(int);
	for (int i = 0;i < arrSize;i++) {
		iarr[i] = 0;
	}
	//	因为分配了内存，使用完需要删除
	delete[] iarr;
	//	新式数组
	std::array<int, arrSize> stdArray;
	std::cout << stdArray[0] << std::endl;
	delete& stdArray;
}

//	通过函数传递字符串时最好通过引用传递，否则会变成复制（分配新内存）
void PrintString(const std::string& str) {
	std::cout << str << std::endl;
}

void initString() {
	//	字符指针，实际是数组，可当做字符串使用，实际字符是分开存储为['l','o','t','a','w','a','y']
	const char* cname = "lotaway";
	//	字符数组，utf8，注意长度是7，比实际字符多1位，因为在最后会自动添加\0作为结束字符
	const char aname[7] = "lotawa";
	//	更长的字符数组
	const wchar_t* cname2 = L"lotaway";
	//	utf16，每个字符2字节
	const char16_t* ccname = u"lotaway";
	//	utf32，每个字符4字节
	const char32_t* cccname = U"lotaway";
	//	获取字符数组长度
	std::cout << strlen(cname) << std::endl;
	//	复制字符数组
	char* cpyName = new char[5];
	//strcpy_s(cpyName, strlen(cpyName - 1), cname);

	//	字符串
	std::string sname = "lotaway";
	std::cout << sname << std::endl;
	//	获取字符串长度
	std::cout << sname.size() << std::endl;
	//	字符串拼接，而不能在定义时直接添加
	sname += "'s name!";
	//	或者在定义时定义多个字符串
	std::string ssname = std::string("lotaway") + "'s name";
	//	判断字符串中是否存在某些子字符串
	bool contains = ssname.find("name") != std::string::npos;

	//	语法糖拼接，需要引入库
	using namespace std::string_literals;
	std::string uname = "lotaway"s + "'s name";
	std::u32string uuname = U"lotaway"s + U"'s name";
	//	可换行字符数组
	const char* change = R"(line1
		line2
		line3
		line4
)";
}

class OnlyReadFn {
private:
	int m_x;
	mutable int getCount;
public:
	OnlyReadFn() : m_x(0), getCount(0) {

	}
	//	使用const在尾部将函数标记为不会修改类
	const int getX() const {
		//	不可用赋值，因为已经标记为const
		//m_x = 2;
		//	但对mutable依旧可以修改,mutable>const>variable
		getCount += 1;
		return m_x;
	}
};

void initConst() {
	//	常量，只读
	const int MAX_AGE = 140;
	//	编译期常量
	constexpr int MAX_COUNT = 1;
	//	常量指针，可修改指针，但不可修改指针指向的内容
	const int* a = new int(2);
	std::cout << *a << std::endl;
	//	不允许修改a指向的内容
	//*a = 2;
	//	允许修改指针本身
	a = &MAX_AGE;
	std::cout << *a << std::endl;
	//	指针常量，不可修改指针，但可修改指针指向的内容
	int* const b = new int;
	std::cout << b << std::endl;
	//	允许修改b指针本身
	*b = MAX_AGE;
	std::cout << b << std::endl;
	//	不允许修改b指针指向的内容
	//b = &MAX_AGE;
	//	常量指针常量？既不可以修改指针，也不可以修改指针指向的内容
	const int* const c = new int(3);
	std::cout << *c << std::endl;
	//*c = MAX_AGE:
	//c = &MAX_AGE;
	OnlyReadFn onlyReadFn;
	std::cout << onlyReadFn.getX() << std::endl;
}

void initLabbda() {
	int a = 8;
	//	lambda表达式，一次性函数，&代表引用，=代表值传递
	auto lam = [&]() {
		a++;
		std::cout << a << std::endl;
		std::cin.get();
	};
	lam();
}

//	重载操作符
class Vec {
public:
	float m_x, m_y;
	Vec(float x, float y) : m_x(x || 0.0f), m_y(y || 0.0f) {}
	Vec add(const Vec& _vec) const {
		return Vec(m_x + _vec.m_x, m_y + _vec.m_y);
	}
	//	重载操作符加号
	Vec operator+(const Vec& _vec) const {
		return this->add(_vec);
	}
	Vec multiply(const Vec& _vec) const {
		return Vec(m_x * _vec.m_x, m_y * _vec.m_y);
	}
	//	重载操作符乘号
	Vec operator*(const Vec& _vec) const {
		return this->multiply(_vec);
	}
	bool isEqual(const Vec& _vec) const {
		return m_x == _vec.m_x && m_y == _vec.m_y;
	}
	//	重载相等操作符
	bool operator==(const Vec& _vec) const {
		return this->isEqual(_vec);
	}
};
//	输出流的<<操作符也可以重载
std::ostream& operator<<(std::ostream& stream, const Vec& _vec) {
	stream << _vec.m_x << ',' << _vec.m_y;
	return stream;
}

void initCalculate() {
	Vec vec(0.0f, 0.0f);
	Vec walkSpeed(1.0f, 1.0f);
	Vec powerUp(2.0f, 2.0f);
	//	调用方法完成计算
	Vec mulRes = vec.add(walkSpeed).multiply(powerUp);
	//	调用操作符完成计算
	Vec mulRes2 = (vec + Vec(walkSpeed)) * powerUp;
	std::cout << (mulRes == mulRes2) << std::endl;
	//	std::ostream << Vec类型的操作符重载了，可以输出Vec类型
	std::cout << mulRes2 << std::endl;
}

int* createArray() {
	//	栈上分配，返回后就会销毁变量，返回值没有意义
	//int arr[50];
	//	堆上分配，会一直保留直到手动摧毁
	int* arr = new int[50];
	return arr;
}
//	利用栈类来摧毁堆类
class Entity {
public:
	void dododo() {}
};
class ScopeEntity {
private:
	Entity* m_entity;
public:
	//	传入堆上的实例
	ScopeEntity(Entity* entity) : m_entity(entity) {}
	//	删除堆上的实例
	~ScopeEntity() {
		delete m_entity;
	}
};

void initStackClass() {
	//	此处有自动隐性转换，相当于ScopeEntity(new Entity())
	{
		ScopeEntity se = new Entity();
	}
	//	离开作用域后，栈上的se自动销毁，而传入的堆上的entity实例也会因为调用折构函数而被删除
}

//	智能指针，自动管理new和delete
void initIntelligencePointer() {
	//	unique_ptr用于创建唯一指针，不可复制
	{
		//	创建唯一指针，离开作用域后自动删除堆上的实例变量
		std::unique_ptr<Entity> uniqueEntity(new Entity());
		//	或者
		std::unique_ptr<Entity> uniqueEntity2 = std::make_unique<Entity>();
		uniqueEntity->dododo();
	}
	//	shared_ptr用于创建共享指针，可复制，当引用归零则自动销毁
	std::shared_ptr<Entity> shareEntity;
	{
		std::shared_ptr<Entity> shareEntity2(new Entity());
		shareEntity = shareEntity2;
	}
	//	依旧可以调用，因为entity0还有引用
	std::cout << shareEntity << std::endl;
	//	weak_ptr用于创建弱共享指针，不会被记入引用中
	std::weak_ptr<Entity> weakEntity;
	{
		std::shared_ptr<Entity> shareEntity3(new Entity());
		weakEntity = shareEntity3;
	}
	//	已经被删除，因为不计入引用
	std::cout << weakEntity.lock() << std::endl;
	std::cin.get();
}

class SS {
private:
	char* m_buffer;
	unsigned int m_size;
public:
	SS(const char* content) {
		m_size = (unsigned int)strlen(content) + 1;
		m_buffer = new char[m_size];
		fri(*this, content);
	}
	//	拷贝时会调用的构造函数
	SS(const SS& ss) : m_size(ss.m_size) {
		m_buffer = new char[m_size];
		//memcpy(m_buffer, ss.m_buffer, ss.m_size);
		fri(*this, ss.m_buffer);
	}
	~SS() {
		delete[] m_buffer;
	}
	void print() const {
		std::cout << m_buffer << std::endl;
	}
	char& operator[](unsigned int index) {
		return m_buffer[index];
	}
	//	友元方法声明，可让私有变量也被外部函数调用
	friend void fri(SS& ss, const char* content);
};

//	友元方法定义，可以调用声明处的实例私有属性
void fri(SS& ss, const char* content) {
	memcpy(ss.m_buffer, content, ss.m_size);
}

void stringCopy() {
	SS ss("lotaway");
	//	浅拷贝，对char* m_buffer只会拷贝指针，最终导致在折构方法中清理内存时报错，因为无法删除两次
	SS sp = ss;
	sp[1] = 'i';
	ss.print();
	sp.print();
	std::cin.get();
}

class Origin {
public:
	void print() const {
		std::cout << "haha" << std::endl;
	}
};

class SpecPointer {
private:
	Origin* origin;
public:
	SpecPointer(Origin* _origin) : origin(_origin) {

	}
	const Origin* operator->() const {
		return origin;
	}
};

void arrowPoint() {
	SpecPointer specPointer = new Origin();
	specPointer->print();
	std::cin.get();
}

struct Vex {
	float x, y, z;
	Vex(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {

	}
};

//	重载输出，方便输出Vex类
std::ostream& operator<<(std::ostream& stream, const Vex& vex) {
	stream << vex.x << ',' << vex.y << ',' << vex.z;
	return stream;
}

void outputVex(const std::vector<Vex>& vexs) {
	//	循环读取项
	for (const Vex& vex : vexs) {
		std::cout << vex << std::endl;
	}
}

//	动态数组
void initVector() {
	//	传入项类型来创建动态数组
	std::vector<Vex> vexs;
	//	随时放入新数据，数组会自动根据长度所需去重新创建新数组（并删除旧数组）
	//	push_back会在当前环境创建一个Vex实例，之后才复制进vector类
	vexs.push_back({ 0.0f, 0.0f, 0.0f });
	vexs.push_back({ 1.0f, 4.0f, 7.0f });
	outputVex(vexs);
	//	清除所有数据
	vexs.clear();
	//	emplace_back会直接在vector内部创建，就不会有先创建再复制导致的效率问题
	vexs.emplace_back(1.0f, 1.0f, 1.0f);
	vexs.emplace_back(2.0f, 7.0f, 8.0f);
	outputVex(vexs);
	//	删除指定索引的值，无法直接用number类型，.begin()相当于开始的0，即删除索引值为1的第二个值
	vexs.erase(vexs.begin() + 1);
	outputVex(vexs);
}

//	用于多返回值的struct
struct Return1 {
	std::string x;
	std::string y;
	int z;
};
//	利用struct多返回值
Return1 returnStruct() {
	return { "hello", "lotaway", 1 };
}

//	用于传递多个引用参数并多返回值
void returnParams(std::string& str1, std::string& str2, int& z) {
	str1 = "hello";
	str2 = "lotaway";
	z = 1;
}

//	用于数组多返回值
std::array<std::string, 2> returnArray() {
	std::array<std::string, 2> arr;
	arr[0] = "hello";
	arr[1] = "lotaway";
	return arr;
}

//	返回自定义的多返回值
std::tuple<std::string, std::string, int> returnTuple() {
	return std::make_tuple("hello", "lotaway", 1);
}

//	多返回值方法：1、struct；2、传递引用参数再赋值；3、返回数组；4、tuple定义多个不同类型值。
void initReturn() {
	auto return1 = returnStruct();
	std::cout << return1.x + ',' + return1.y + ',' + std::to_string(return1.z) << std::endl;
	std::string str1, str2;
	int z;
	returnParams(str1, str2, z);
	std::cout << str1 + ',' + str2 + ',' + std::to_string(z) << std::endl;
	//returnParams(nullptr, nullptr, z);
	auto array = returnArray();
	std::cout << array[0] + ',' + array[1] << std::endl;
	std::tie(str1, str2, z) = returnTuple();
	std::cout << str1 + ',' + str2 + ',' + std::to_string(z) << std::endl;
}

//	template可以通过指定泛型来减少无谓的函数重载定义
template<typename FirstParam>
void template1(FirstParam param) {
	std::cout << param << std::endl;
}

//	template定义类里的变量类型和数组大小
template<typename Arr, int size>
class SArray {
private:
	Arr arr[size];
public:
	int getSize() const {
		return sizeof(arr);
	}
};

//	template类似一种元编程，即代码本身不是在编译时确定，而是在运行期才确定
void initTemplate() {
	template1("hahah");	//	可以自动根据输入值推断类型，或者手动限制template1<std::string>("hahah");
	SArray<int, 5> sarray;
	std::cout << sarray.getSize() << std::endl;
}