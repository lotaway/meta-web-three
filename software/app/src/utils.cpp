#pragma once
#include "./include/stdafx.h"
//	std::mutex 互斥锁
#include <mutex>
//	std::async 异步任务
#include <future>
//	静态库，需要打包到exe文件中，效率更好，但单文件更大
//	动态库，一般是放置到exe文件旁边
//	引入依赖库
//#include <GLFW/glfw3.h>
//	引入解决方案中的其他项目
// emsdk无法识别，只能使用引号加相对路径"../../engine/src/engine.h"，除了cpp和标准库以外的文件都没有被编译进去wasm
#include <engine.h>
#include "logger.h"
#include "utils.h"

#define debugger(msg) std::cout << "main::debugger:" + msg << std::endl

extern int g_variable;	//	将从外部所有文件里找到这个全局值，目前定义在log.cpp里

namespace utils {
	//	来源于静态库或动态库的方法
	void useLibrary() {
		//int a = glfwInit();
		//std::cout << a << std::endl;
		engine::printMessage();
	}

	void variableAndLog() {
		logger::out("hello world");
		const char* name = "c++";
		std::string s = "This's string";
		//std::wstring s = "这就是字符串了";
		logger::out("welcome", name);
		int value = -1;
		unsigned int uValue = 2;
		float fVal = 5.5f;
		double dVal = 5.5;
		short sVal = 1;
		long lVal = 2;
		long int liVal = 3;
		bool needMoreInfo = value < 0;
		//logger.out("going end", value);
		std::cin.get();

		if (needMoreInfo) {
			logger::out("need more info");
			for (int i = 0;i < 10;i++) {
				value--;
				logger::out("value", value);
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
		logger::out("incrementWithPointer b", *b);
		logger::out("what about a", a);
		incrementWithPointer(&a);
		logger::out("incrementWithPointer a", a);
		logger::out("what about b", *b);
		incrementWithReference(a);
		logger::out("incrementWithReference a", a);
		std::cin.get();

		int& ref = a;
		//	ref是对a的引用，将c这个值赋值给ref和a，即a=8，ref=8，并且c=8，两者互不干扰
		ref = c;
		logger::out("ref", ref);
		//	指针可以被修改指向到新的地址，但引用不可以，只能赋值
		b = &c;
		c = 10;
		logger::out("b change", *b);
	}

	void localStaticVar() {
		//	局部静态变量实际上是持续存在的静态变量，第一次调用会初始化（值为1），之后保持增长（2、3、4...）
		static int variable = 1;
		variable++;
		std::cout << "local static variable " + std::to_string(variable) << std::endl;
	}

	void initStatic() {
		logger::out("extern global variable", g_variable);
		//	调用多次函数内将输出不同的局部静态变量
		localStaticVar();	//	2
		localStaticVar();	//	3
		localStaticVar();	//	4
	}

	//	传统枚举，枚举名只是作为类型而非命令空间存在，会自动将内部属性名称作为当前作用域变量使用
	enum PlayerLevel {
		PlayerLevel_EntryLevel,
		PlayerLevel_Intermediate,
		PlayerLevel_Expert
	};

	//	枚举类，枚举名同时作为类型和命名空间存在，需在前缀加上类名才能调用
	enum class PlayerStatus {
		Enabled,
		Disabled
	};

	Player::Player(PlayerLevel level = PlayerLevel_EntryLevel) : m_level(level), m_status(PlayerStatus::Enabled), m_positionX(0), m_positionY(0), m_speed(15) {
	}

	Player::~Player() {
		std::cout << "go out" << std::endl;
	}

	void Player::move(int new_x, int new_y) {
		m_positionX += new_x * m_speed;
		m_positionY += new_y * m_speed;
	}

	void fastMove(Player& player, int new_x, int new_y) {
		player.m_positionX += new_x * player.m_speed * 2;
		player.m_positionY += new_y * player.m_speed * 2;
	}

	void NormalPerson::move(int new_x, int new_y) {
		m_positionX += new_x * m_speed;
		m_positionY += new_y * m_speed;
	}
	void NormalPerson::follow(Player& _player) {
		m_like = &_player;
	}

	Trainer::Trainer(int runNumber, int age, int sex) {
		m_runLevel = 0;
		m_runNumber = runNumber;
		m_age = age;
		m_sex = sex;
	}

	void Runner::run() {};

	Racer::Racer(const char& cup, int rank) : m_cup(cup), m_rank(rank) {}

	void Racer::run() {
		std::cout << "run" << std::endl;
	}
	
	std::string Winner::getNews() {
		return "He win.";
	}

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
			iarr[i] = i;
		}
		std::cout << *iarr << std::endl;
		//	因为在堆上分配了内存，使用完需要删除
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

	OnlyReadFn::OnlyReadFn() : m_x(0), getCount(0) {

	}

	const int OnlyReadFn::getX() const {
		//	不可用赋值，因为已经标记为const
		//m_x = 2;
		//	但对mutable依旧可以修改,mutable>const>variable
		getCount += 1;
		return m_x;
	}

	void initConst() {
		//	常量，只读
		const int MAX_AGE = 140;
		//	编译期常量
		constexpr int MAX_COUNT = 1;
		//	常量指针，即指向常量的指针。可修改指针，但不可修改指针指向的内容
		const int* a = new int(2);
		std::cout << *a << std::endl;
		//	不允许修改a指向的内容
		//*a = 2;
		//	允许修改指针本身
		a = &MAX_AGE;
		std::cout << *a << std::endl;
		//	指针常量，即指针是常量。不可修改指针，但可修改指针指向的内容
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

	Vec::Vec(float x, float y): m_x(x || 0.0f), m_y(y || 0.0f) {}

	Vec Vec::add(const Vec& _vec) const {
		return Vec(m_x + _vec.m_x, m_y + _vec.m_y);
	}

	Vec Vec::operator+(const Vec& _vec) const {
		return this->add(_vec);
	}

	Vec Vec::multiply(const Vec& _vec) const {
		return Vec(m_x * _vec.m_x, m_y * _vec.m_y);
	}

	Vec Vec::operator*(const Vec& _vec) const {
		return this->multiply(_vec);
	}

	bool Vec::isEqual(const Vec& _vec) const {
		return m_x == _vec.m_x && m_y == _vec.m_y;
	}

	bool Vec::operator==(const Vec& _vec) const {
		return this->isEqual(_vec);
	}

	//	输出流的<<操作符也可以重载
	std::ostream& operator<<(std::ostream& stream, const Vec& _vec) {
		stream << _vec.m_x << ',' << _vec.m_y;
		return stream;
	}

	Vec& Vecv::getVec() {
		return vec;
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
	};

	int* createArray() {
		//	栈上分配，返回后就会销毁变量，返回值没有意义
		//int arr[50];
		//	堆上分配，会一直保留直到手动摧毁
		int* arr = new int[50];
		return arr;
	}

	void Entity::dododo() {}

	ScopeEntity::ScopeEntity(Entity* entity) : m_entity(entity) {}

	ScopeEntity::~ScopeEntity() {
		delete m_entity;
	}

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
	//  现代C++《生产环境都使用智能指针而非原始指针，单纯只是学习和积累经验则使用原始指针，甚至自己定制智能指针。

	SS::SS(const char* content) {
		m_size = (unsigned int)strlen(content) + 1;
		m_buffer = new char[m_size];
		fri(*this, content);
	}

	SS::SS(const SS& ss) : m_size(ss.m_size) {
		m_buffer = new char[m_size];
		//memcpy(m_buffer, ss.m_buffer, ss.m_size);
		fri(*this, ss.m_buffer);
	}

	SS::~SS() {
		delete[] m_buffer;
	}

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

	void Origin::print() const {
		std::cout << "haha" << std::endl;
	}

	SpecPointer::SpecPointer(Origin* _origin) : origin(_origin) {

	}
	
	const Origin* SpecPointer::operator->() const {
		return origin;
	}

	void arrowPoint() {
		SpecPointer specPointer = new Origin();
		specPointer->print();
		std::cin.get();
	}

	Vex::Vex(float _x, float _y) : x(_x), y(_y) {
	};

	void initUnion() {
		using namespace std;
		Vex4 v4 = { 1.0f, 2.0f, 3.0f, 4.0f };
		//	可以当作2个Vex类型输出，也可以当作4个浮点类型输出，灵活且共享内存
		cout << v4.a.x << ',' << v4.a.y << ',' << v4.b.x << ',' << v4.b.y << endl;
		cout << v4.p1 << ',' << v4.p2 << ',' << v4.p3 << ',' << v4.p4 << endl;
	}

	//	重载输出，方便输出Vex类
	std::ostream& operator<<(std::ostream& stream, const Vex& vex) {
		stream << vex.x << ',' << vex.y;
		return stream;
	}

	template<typename Vec>
	void outputVex(const std::vector<Vec>& vexs) {
		//	循环读取项
		for (const Vec& vex : vexs) {
			std::cout << vex << std::endl;
		}
	}

	//	动态数组
	void initVector() {
		//	传入项类型来创建动态数组
		std::vector<Vex> vexs;
		//	随时放入新数据，数组会自动根据长度所需去重新创建新数组（并删除旧数组）
		//	push_back会在当前环境创建一个Vex实例，之后才复制进vector类
		vexs.push_back({ 0.0f, 0.0f });
		vexs.push_back({ 1.0f, 4.0f });
		outputVex<Vex>(vexs);
		//	清除所有数据
		vexs.clear();
		//	emplace_back会直接在vector内部创建，就不会有先创建再复制导致的效率问题
		vexs.emplace_back(1.0f, 1.0f);
		vexs.emplace_back(2.0f, 7.0f);
		outputVex<Vex>(vexs);
		//	删除指定索引的值，无法直接用number类型，.begin()相当于开始的0，即删除索引值为1的第二个值
		vexs.erase(vexs.begin() + 1);
		outputVex(vexs);
	}

	Return1 returnStruct() {
		return { "hello", "lotaway", 1 };
	}

	void returnParams(std::string& str1, std::string& str2, int& z) {
		str1 = "hello";
		str2 = "lotaway";
		z = 1;
	}

	std::array<std::string, 2> returnArray() {
		std::array<std::string, 2> arr;
		arr[0] = "hello";
		arr[1] = "lotaway";
		return arr;
	}

	std::tuple<std::string, std::string, int> returnTuple() {
		return std::make_tuple("hello", "lotaway", 1);
	}

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
		auto [str3, str4, z1] = returnTuple();
		std::cout << str1 + ',' + str2 + ',' + std::to_string(z) << std::endl;
	}

	template<typename FirstParam>
	void template1(FirstParam param) {
		std::cout << param << std::endl;
	};

	template<typename Arr, int size>
	int SArray<Arr, size>::getSize() const {
		return sizeof(arr);
	}

	//	template类似一种元编程，即代码本身不是在编译时确定，而是在运行期才确定
	void initTemplate() {
		template1("hahah");	//	可以自动根据输入值推断类型，或者手动限制template1<std::string>("hahah");
		SArray<int, 5> sarray;
		std::cout << sarray.getSize() << std::endl;
	}

	template<typename Value>
	//	如果形参里定义的回调函数是匿名类型会导致lambda无法使用[]捕获作用域变量，会报错参数不符合
	//void each(const std::vector<Value>& values, void(*handler)(Value)) {
	//	形参里用标准库方法模板定义回调函数类型，lambda才能使用[]捕获作用域变量
	void each(const std::vector<Value>& values, const std::function<void(Value)>& handler) {
		for (Value value : values) {
			handler(value);
		}
	}

	void initLambda() {
		const char* name = "extra";
		using Value = int;
		std::vector<Value> vec = { 1, 2, 3 };
		// 匿名函数里没有当前作用域的变量
		each<Value>(vec, [](Value val) { logger::out("name", val); });
		// 匿名函数里需要有当前作用域的所有变量
		each<Value>(vec, [=](Value val) { logger::out(name, val); });
		// 匿名函数里需要有当前作用域的某个变量
		each<Value>(vec, [&name](Value val) { logger::out(name, val); });
	}

	void initAuto() {
		std::vector<int> vec = { 1, 2, 3 };
		auto it = std::find_if(vec.begin(), vec.end(), [](int val) { return val > 2; });
		logger::out(*it);
	}

	bool isFinish = false;

	void doWork() {
		std::cout << std::this_thread::get_id() << std::endl;
		using namespace std::literals::chrono_literals;
		while (!isFinish) {
			logger::out("pending");
			std::this_thread::sleep_for(1s);
		}
	}

	void initThread() {
		PROFILE_FUNCTION();
		std::cout << std::this_thread::get_id() << std::endl;
		std::thread worker(doWork);
		isFinish = true;
		worker.join();
		logger::out("线程结束");
	}

	void initSort() {
		std::vector<int> vecInt = { 1, 5, 7, 3 , 2 };
		//  内置排序方法，默认按照从小到大排序，如果没有传递判断方法的话
		std::sort(vecInt.begin(), vecInt.end(), [](int a, int b) {
			return a < b;
			});
		utils::outputVex<int>(vecInt);
	}

	//	类型转换依赖于RTTI(Runing Time Type Info，运行时类型信息），启用该配置和使用C++函数风格类型转转
	void initTypeConvertionCheck() {
		double a = 5.25;
		int b1 = a;	//	隐式转换，C风格
		int b2 = (double)a;	//	显式转换，C风格
		//	以下是C++风格的类型转换，实际上是一个函数，有额外开销：
		//	静态类型转换，方便编译器在编译和运行阶段确定错误哦，并且有利于后续维护查找哪里进行了类型转换
		int b = static_cast<int>(a);
		//	类型双关的转换
		logger::timer timer("initTypeConvertionCheck repinterpret_cast");
		double* t = (double*)reinterpret_cast<logger::timer*>(&timer);
		//	动态类型转换，适用于确认继承关系下的类型
		Racer* racer = new Racer(*"worker", 1);
		Runner* runner = racer;
		Racer* newRacer = dynamic_cast<Racer*>(runner);
		//	将子类指针赋值给父类指针变量后，可通过动态类型转换确认是否为某特定子类类型，转换后若有值则是该子类类型，失败Null则不是
		if (newRacer) {

		}

		//	常量与变量的转换
		const char* cc = "hello";
		char* dd = const_cast<char*>(cc);
	}

	//	读取可能不存在的文件，并设置为可选返回值
	std::optional<std::string> readFileAsString(const std::string filePath) {
		std::fstream fstream(filePath);
		if (fstream) {
			//	read file(no done)
			std::string line;
			std::string result;
			while (std::getline(fstream, line)) {
				result += line;
			}
			fstream.close();
			return result;
		}
		return {};
	}

	void initGetFile() {
		auto file = readFileAsString("file.json");
		//	method 1, check data if exist
		if (file.has_value()) {
			logger::out(file.value());
		}
		else {
			logger::out("File No Found!");
		}
		//	method 2, if data not exist, return value_or value;
		std::string data = file.value_or("File No Found!");
		logger::out(data);
	}

	//	指定多类型的单一变量variant，相比union更注重类型安全，但union更具效率和更少内存占用
	void sigleVariantMultiType() {
		std::variant<std::string, int> data;
		//	可能是字符串
		data = "lotaway";
		//	需要指定获取的类型
		logger::out(std::get<std::string>(data));
		//	也可能是数值
		data = 30;
		logger::out(std::get<int>(data));
		//	无法确定最终类型的情况下，最好是通过判断获取
		logger::out(*std::get_if<std::string>(&data));
	}

	//	std::any 任意类型的单一变量（不推荐用），相比variant无需事先声明所有可能的类型，相同的点是在取值时需要指定类型，缺点是需要动态分配内存导致性能问题
	void anyValue() {
		std::any data;
		data = "lotaway";
		data = 30;
		logger::out(std::any_cast<int>(data));
	}

	//	通过异步（多线程）并行处理任务，提升性能

	namespace Hazel {

		class Mesh {
		public:
			Mesh(const std::string& _filepath) : filepath(_filepath) {}
			static Mesh* load(const std::string& filepath) {
				//	 do something...
				Mesh* mesh = new Mesh(filepath);
				return mesh;
			}
		private:
			const std::string& filepath;
		};

		template<class T>
		struct Ref {
		public:
			using _TargetType = T;
			Ref(_TargetType* _t) : t(_t) {}
			~Ref() {
				delete t;
			}
		private:
			_TargetType* t;
		};
		//	互斥锁
		static std::mutex s_meshesMutex;

		class EditorLayer {
		public:

			static void loadMesh(std::vector<Ref<Mesh>>* meshes, std::string filepath) {
				auto mesh = Mesh::load(filepath);
				std::lock_guard<std::mutex> lock(s_meshesMutex);
				meshes->push_back(mesh);
			}

			void loadMeshes() {
				std::ifstream stream("src/Models.txt");
				std::string line;
				std::vector<std::string> meshFilepaths;
				while (std::getline(stream, line))
					meshFilepaths.push_back(line);
#define ASYNC 1
#if ASYNC
				for (const auto& file : meshFilepaths)
					m_futures.push_back(std::async(std::launch::async, loadMesh, &m_meshes, file));
#else
				for (const auto& file : meshFilepaths)
					m_meshes.push_back(Mesh::load(file));
#endif
			}
		private:
			std::vector<Ref<Mesh>> m_meshes;
			std::vector<std::future<void>> m_futures;
		};

		void initLockAndAsync() {
			EditorLayer editorLayer;
			editorLayer.loadMeshes();
		}
	};

	//	字符串优化：最好是减少使用string而用char，子字符串用string_view
	void initStringOptimization() {
		//	bad, 4 times memory allocate, 
		std::string str = "Way Luk";
		logger::out(str);
		std::string first = str.substr(0, 3);
		logger::out(first);
		std::string last = str.substr(3, 7);
		logger::out(last);
		//	good, but still need string, 1 time memory allocate
		std::string_view firstName1(str.c_str(), 3);
		std::cout << firstName1 << std::endl;
		std::string_view lastName1(str.c_str() + 4, 3);
		std::cout << lastName1 << std::endl;
		//	actually good, only need char, 
		const char* name = "Way Luk";
		logger::out(name);
		std::string_view firstName2(name, 3);
		std::cout << firstName2 << std::endl;
		std::string_view lastName2(name + 4, 3);
		std::cout << lastName2 << std::endl;
	}
	//	实际上Release环境编译时会有SSO小字符串优化，会自动将字符不多的字符串用栈缓冲区而非堆去分配内存，只有比较长的字符才会正常用堆分配内存，在VS2019中，触发这种机制的长度是15个字符。

	//	设计模式：单例模式
	class Random {
	public:
		static Random& get() {
			return s_instance;
		}
		static float Float() {
			return get()._Float();
		}
		//	ban copy instance
		Random(const Random&) = delete;
	private:
		static Random s_instance;
		float s_randomGenerator = 0.5f;
		float _Float() {
			return s_randomGenerator;
		}
		Random() {}
	};

	Random Random::s_instance;

	//	左值和右值，有具体位置的是左值，只是临时值的是右值，右值没有位置，所以不能被赋值
	int getValue() {
		return 10;
	}
	void setValue(int val) {
	}
	//	引用时，只能传递左值，不能传递右值
	void setValue(std::string& name) {

	}
	//	常量引用时，可以传递左值和右值
	void setValue2(const std::string& name) {

	}
	//	双重引用时，只能传递右值，不能传递左值
	void setValue3(std::string&& name) {

	}
	void initLValueAndRValue() {
		//	这里a是左值，1是右值
		int a = 1;
		//	这里a和b都是左值
		int b = a;
		//	这里c是左值，getValue()返回一个右值
		int c = getValue();
		//	不能堆getValue()赋值，因为它返回一个右值
		//getValue() = a;
		//	这里a是左值
		setValue(a);
		//	这里2是右值
		setValue(2);
		
		//	firstName是左值，"Way"是右值
		std::string firstName = "Way";
		//	lastName是右值，"Luk"是右值
		std::string lastName = "Luk";
		//	这里右边是两个左值firstName和lastName，但firstName + lastName加起来生成了一个临时值，所以是右值
		std::string fullName = firstName + lastName;

		//	引用时，只能传递左值
		setValue(fullName);
		//	引用时，不能传递右值，此处是常量字面量右值
		//setValue("lotaway");
		// 
		//	常量引用时，既能传递左值
		setValue2(fullName);
		//	常量引用时，也能传递右值
		setValue2("Way Luk");

		//	双重引用时，只能传递右值
		setValue3("Way Luk");
		//	双重引用时，不能传递左值
		//serValue3(fullName);

		//	可以利用这种传递特性写重载方法，完成移动语义，当传递的是右值时，可以放心进行使用甚至修改，因为只是临时使用而不会复制或者影响其他内容。
	}

	//	移动语义
	class String {
	public:
		String() = default;
		String(const char* data) {
			printf("Created!\n");
			m_size = strlen(data);
			m_data = new char[m_size];
			memcpy(m_data, data, m_size);
		}
		String(const String& other) {
			printf("Copyed!\n");
			m_size = other.m_size;
			m_data = new char[m_size];
			memcpy(m_data, other.m_data, m_size);
		}
		String(String&& other) noexcept {
			printf("Moved!\n");
			m_size = other.m_size;
			//	在双重引用下能拿到右值，并简单地移动指针，之后删除原指针
			m_data = other.m_data;
			other.m_data = nullptr;
		}
		~String() {
			delete m_data;
		}
	private:
		size_t m_size;
		char* m_data;
	};

	class Enstring {
	public:
		Enstring(const String& name) : m_name(name) {}
		//	形参里name是右值，但放到初始化列表里的实参name将作为左值传入，需要使用std;:move无条件再次强制变成右值
		Enstring(String&& name): m_name(std::move(name)) {}
	private:
		String m_name;
	};

	void initStringAndMove() {
		//	如果不使用双重引用&&重载String构造方法，这个创建方式会先在堆上分配创建String("lotaway")，之后在Enstring通过m_name初始化列表又重新在堆上分配创建（即使使用的是引用&，但String的构造创建与复制方法决定了会分配成两个堆）
		Enstring enstring("lotaway");
	}

	// iteralor迭代器，使用指针进行循环迭代取值
	void initIteralor() {

		//	vector iterator
		std::vector<int> values = { 1, 2, 3, 4, 5 };
		//	method 1: using trandition for
		for (int i = 0;i < values.size(); i++) {
			logger::out(values[i]);
		}
		//	method 2: using for :
		for (int value : values) {
			logger::out(value);
		}
		// method 3: vector inset iterator
		for (std::vector<int>::iterator iterator = values.begin(); iterator != values.end(); iterator++) {
			logger::out(*iterator);
		}


		//	map iterator
		using ScoreMap = std::unordered_map<std::string, int>;
		using SMConstIter = ScoreMap::const_iterator;
		ScoreMap map;
		map["lotaway"] = 30;
		map["C Plus Plus"] = 100;
		//	method 1: using unordered_map::const_iterator
		for (ScoreMap::const_iterator it = map.begin(); it != map.end(); it++) {
			//	it is a vector:pair
			auto& key = it->first;
			auto& value = it->second;
			logger::out(key, value);
		}
		//	method 2: using for :
		for (auto& it : map) {
			auto& key = it.first;
			auto& value = it.second;
		}
		//	method 3: using for : with 结构解构
		for (auto [key, value] : map) {
			logger::out(key, value);
		}
	}

	//	make you own vector iterator
	template<typename _Vector>
	class MyVectorIterator {
	public:
		using Iterator = MyVectorIterator;
		using ValueType = typename _Vector::ValueType;
		using PointerType = ValueType*;
		using ReferenceType = ValueType&;
		MyVectorIterator(PointerType ptr) : m_ptr(ptr) {

		}
		Iterator& operator++() {
			m_ptr++;
			return *this;
		}
		Iterator operator++(int) {
			Iterator iterator = *this;
			++(*this);
			return iterator;
		}
		Iterator operator--() {
			m_ptr--;
			return *this;
		}
		Iterator operator--(int) {
			Iterator iterator = *this;
			--(*this);
			return iterator;
		}
		ReferenceType operator[](int index) {
			return *(m_ptr + index);
		}
		PointerType operator->() {
			return m_ptr;
		}
		PointerType operator*() {
			return *m_ptr;
		}
		bool operator==(const Iterator& other) const {
			return m_ptr == other.m_ptr;
		}
	private:
		PointerType m_ptr;
	};

	template<typename T, size_t S = 2>
	class MyVector {
	public:
		using ValueType = T;
		using Iterator = MyVectorIterator<MyVector>;
		MyVector() {
			logger::out("MyVector created");
			static_assert(S > 0, "size_t S should not be 0");
			reAlloc(S);
		}
		~MyVector() {
			logger::out("MyVector destroyed");
			clear();
			::operator delete(m_data, m_capacity * sizeof(T));
		}
		size_t size() const {
			return m_size;
		}
		void push_back(const T& value) {
			if (m_size >= m_capacity)
				reAlloc(m_capacity + m_capacity / 2);
			m_data[m_size++] = value;
		}
		void push_back(T&& value) {
			if (m_size >= m_capacity)
				reAlloc(m_capacity + m_capacity / 2);
			m_data[m_size] = std::move(value);
			m_size++;
			//value = nullptr;
		}
		template<typename... Args>
		T& emplace_back(Args&... args) {
			if (m_size >= m_capacity)
				reAlloc(m_capacity + m_capacity / 2);
			new (&m_data[m_size]) T(std::forward<Args>(args)...);
			return m_data[m_size++];
		}
		void pop_back() {
			if (m_size > 0) {
				m_size--;
				m_data[m_size].~T();
			}
		}
		void clear() {
			for (size_t i = 0; i < m_size; i++)
				m_data[i].~T();
			m_size = 0;
		}
		T& operator[](size_t index) {
			return m_data[index];
		}
		const T& operator[](size_t index) const {
			return m_data[index];
		}
		Iterator begin() {
			return Iterator(m_data);
		}
		Iterator end() {
			return Iterator(m_data + m_size);
		}
	private:
		T* m_data = nullptr;
		size_t m_capacity = 0;
		size_t m_size = 0;
		void reAlloc(size_t newCapacity) {
			T* newBlock = (T*)::operator new(newCapacity * sizeof(T));
			if (newCapacity < m_size)
				m_size = newCapacity;
			for (size_t i = 0; i < m_size; i++)
				new (&newBlock[i]) T(std::move(m_data[i]));
			for (size_t i = 0; i < m_size; i++)
				m_data[i].~T();
			::operator delete(m_data, m_capacity * sizeof(T));
			m_data = newBlock;
			m_capacity = newCapacity;
		}
	};

	void printMyVector(const MyVector<std::string>& vector) {
		logger::out(vector.size());
		for (int i = 0; i < vector.size(); i++)
			std::cout << "vector item: " + vector[i] << std::endl;
	}

	void initMyVector() {
		MyVector<std::string> myVector;
		myVector.emplace_back("heihei");
		myVector.push_back("haha");
		printMyVector(myVector);
		//myVector.push_back("way luk");
		myVector.emplace_back("lotaway");
		printMyVector(myVector);
		myVector.pop_back();
		printMyVector(myVector);
	}

	void initCustomIterator() {
		MyVector<int> mVector;
	}

	//	双重检查锁？

	// hash map 官方实现：unordered_map，无序且键名唯一，随机桶分配方式
	void initHashMap() {
		//	需要提供键名类型和键值类型才能创建hashmap
		std::unordered_map<std::string, int> umap;

		//	添加数据（若原本的键已经存在则覆盖）
		umap.emplace("wayluk", 30);
		umap.emplace("lotaway", 18);

		//	获取数据
		std::unordered_map<std::string, int>::const_iterator it = umap.find("wayluk");
		if (it == umap.end())
			std::cout << "找不到数据" << std::endl;
		else
			std::cout << "找到了！！：" << it->second << std::endl;

		//	删除数据，此处为单条删除，也可以迭代器范围性删除
		umap.erase("wayluk");

		//	插入新数据（只会插入新键不会覆盖已有，且允许批量插入多个值或其他map）
		//	方式1：传递pair
		std::pair<std::string, int> newGuy("wayluk", 31);
		umap.insert(newGuy);
		//	方式2：传入另一个map（一部分或者全部）
		std::unordered_map<std::string, int> other_map{ { "mimi", 27 }, {"haha", 37} };
		umap.insert(other_map.begin(), other_map.end());
		//	方式3：直接字面量初始化
		umap.insert({ { "shutup", 30 }, { "hate", 30 }, {"eatshit", 30} });

		//	迭代器范围性删除
		if (!umap.empty())
			umap.erase(other_map.begin(), other_map.end());

		//	循环输出
		for (auto& m : umap)
			std::cout << m.first << ":" << m.second << std::endl;
	}

	//	用hashmap完成在数组里找到指定结果的两数相加，如提供数组[30,40,60,80]和总值100，其中40+60=100，要求找到40和60并返回它们的索引值
	std::vector<int> getSumTwoBySum(const std::vector<int>& arr, const int sum) {
		std::unordered_map<int, int> requireUMap;
		for (int i = 0, l = arr.size(); i < l; i++) {
			//	当找到和所需的数值一致时，则代表当前这个数值和已在map的索引所在的数值能作为一对，得到结果。
			std::unordered_map<int, int>::const_iterator it = requireUMap.find(sum - arr[i]);
			if (it != requireUMap.end())
				return { it->second, i };
			//	将当前所需要的数值和索引值记录下来
			requireUMap.emplace(arr[i], i);
		}
		return { 0, 0 };
	}

	//	判断输入的数字是否为回文格式，如121，12321就是回文，从高位到低位反过来也是相等，注意不要使用字符串方式
	bool isPalindrome(int x) {
		if (x < 0)
			return false;
		if (x == 0)
			return true;
		const size_t size = static_cast<size_t>(std::log10(x)) + 1;
		if (size == 1)
			return true;
		//int arr[size];
		std::vector<int> arr(size);
		int index = 0;
		size_t max = size;
		while (max > 1) {
			int pos = static_cast<int>(pow(10, --max));
			arr[index++] = x / pos;
			// arr.push_back(x / pos);
			x %= pos;
		}
		arr[index] = x;
		// arr.push_back(x);
		double l = (size - 1) * 0.5;
		for (int i = 0; i < l; i++) {
			if (arr[i] != arr[size - 1 - i])
				return false;
		}
		return true;
	}

	void reverseString(std::string x) {
		std::string originStr = "12321";

		//	反转字符串，方式1，需要创建另外一个字符串
		std::string reStr(originStr.rbegin(), originStr.rend());

		//	反转字符串，方式2，直接修改原有的字符串
		std::reverse(originStr.begin(), originStr.end());

		//	反转字符串，方式3，复制一个字符串进行反转
		std::string cpStr;
		std::reverse_copy(std::begin(originStr), std::end(originStr), std::begin(cpStr));
	}

	struct ListNode {
		int val;
		ListNode* next;
		ListNode() : val(0), next(nullptr) {}
		ListNode(int x) : val(x), next(nullptr) {}
		ListNode(int x, ListNode* next) : val(x), next(next) {}
	};

	ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
		int up = 0;
		int sum = l1->val + l2->val + up;
		ListNode* ln = new ListNode{ sum % 10 };
		ListNode* tempLn = ln;
		up = sum / 10;
		while (l1->next != nullptr || l2->next != nullptr || up > 0) {
			l1 = l1->next == nullptr ? new ListNode() : l1->next;
			l2 = l2->next == nullptr ? new ListNode() : l2->next;
			sum = l1->val + l2->val + up;
			up = sum / 10;
			tempLn->next = new ListNode(sum % 10);
			tempLn = tempLn->next;
		}
		return ln;
	}

	template<size_t size>
	ListNode* createListNodeWithArray(int i_arr[size]) {
		ListNode* ln = new ListNode{ i_arr[0] };
		ListNode* tempLn = ln;
		for (int i = 1; i < size; i++) {
			tempLn->next = new ListNode{ i_arr[i] };
			tempLn = tempLn->next;
		}
		return ln;
	}

	void initListNumberAdd() {
		int i1[1]{ 9 };
		int i2[10]{ 1,9,9,9,9,9,9,9,9,9 };
		ListNode* l1 = createListNodeWithArray<1>(i1);
		ListNode* l2 = createListNodeWithArray<10>(i2);
		ListNode* ln = addTwoNumbers(l1, l2);
		std::cout << ln << std::endl;
	}

	size_t max(size_t a, size_t b) {
		return a > b ? a : b;
	}

	//	获取字符串里梅没有重复字符的子字符串长度，如abcacd中，abc是最长的无重复字符的子字符串，长度为3
	int lengthOfLongestSubstring(std::string s) {
		size_t size = s.size();
		size_t longestCount = 0;
		int startIndex = 0;
		std::unordered_map<char, int> char2LastIndex;
		for (size_t i = startIndex; i < size; i++) {
			char c = s[i];
			auto charLastIndex = char2LastIndex.find(c);
			bool isLast = i == size - 1;
			bool hasRepeat = charLastIndex != char2LastIndex.end() && charLastIndex->second >= startIndex;
			//  if already have a repeat char inside the child string, just count get the longest count and reset the child start index
			if (isLast || hasRepeat) {
				longestCount = max(i - startIndex + (hasRepeat ? 0 : 1), longestCount);
				if (isLast)
					break;
				startIndex = charLastIndex->second + 1;
				if (size - startIndex <= longestCount)
					break;
			}
			char2LastIndex[c] = i;
		}
		return longestCount;
	}

	void quickSort(int arr[], int start, int end) {
		if (start >= end) return;
		int keyVal = arr[start];
		int _start = start;
		int _end = end;
		while (_start < _end) {
			while (_start < _end) {
				if (arr[_end] < keyVal) {
					break;
				}
				_end--;
			}
			while (_start < _end) {
				if (arr[_start] > keyVal) {
					int temp = arr[_start];
					arr[_start] = arr[_end];
					arr[_end] = temp;
					break;
				}
				_start++;
			}
		}
		if (_start == _end)
			arr[_start] = keyVal;
		quickSort(arr, start, _start - 1);
		quickSort(arr, _start + 1, end);
	}

	void testQuickSort() {
		int arr[11] = { 5,6,3,2,7,8,9,1,4,0,0 };
		quickSort(arr, 0, 10);
		for (int x : arr) {
			std::cout << x << " ";
		}
	}

	double getMid(double prevNum, double nextNum, bool isQ = false) {
		if (isQ)
			return nextNum;
		return (prevNum + nextNum) / 2.0f;
	}

	double findMedianSortedArrays(std::vector<int>& nums1, std::vector<int>& nums2) {
		int sumSize = nums1.size() + nums2.size(), neededSize = sumSize / 2 + 1, isQ = sumSize % 2 == 1, index = 0;
		double mid = 0.0;
		std::vector<int>::const_iterator ptr1 = nums1.begin(), ptr2 = nums2.begin();
		while (true) {
			if (ptr1 == nums1.end()) {
				if (index == neededSize - 1) {
					if (!isQ) {
						mid = (mid + *ptr2) / 2.0f;
					}
					else {
						mid = *ptr2;
					}
				}
				else {
					mid = getMid(*(ptr2 + neededSize - 2 - index), *(ptr2 + neededSize - 1 - index), isQ);
				}
				break;
			}
			if (ptr2 == nums2.end()) {
				if (index == neededSize - 1) {
					if (!isQ) {
						mid = (mid + *ptr1) / 2.0f;
					}
					else {
						mid = *ptr1;
					}
				}
				else {
					mid = getMid(*(ptr1 + neededSize - 2 - index), *(ptr1 + neededSize - 1 - index), isQ);
				}
				break;
			}
			if (*ptr1 < *ptr2) {
				if (index == neededSize - 1) {
					mid = getMid(mid, *ptr1, isQ);
					break;
				}
				mid = *ptr1;
				ptr1++;
			}
			else {
				if (index == neededSize - 1) {
					mid = getMid(mid, *ptr2, isQ);
					break;
				}
				mid = *ptr2;
				ptr2++;
			}
			index++;
		}
		return mid;
	}

	void initFindMedianSortedArrays() {
		std::set<std::vector<int>*> uset;
		std::vector<int> f1[]{ {1,2},{3,4} }, f2[]{ {1,3},{2} }, f3[]{ {}, { 1,2,3,4,5 } };
		uset.insert(f1);
		uset.insert(f2);
		uset.insert(f3);
		//std::vector<int> nums1{ 1, 2 }, nums2{ 3, 4 };
		//std::vector<int> nums1{ 1,3 }, nums2{ 2 };
		//std::vector<int> nums1, nums2 = { 1,2,3,4,5 };
		//	output as fixed float 0.000000
		std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
		for (std::vector<int>* it : uset) {
			double mid = findMedianSortedArrays(it[0], it[1]);
			for (int i = 0; i < it->size(); i++) {
				std::cout << '{';
				for (int integer : it[i]) {
					std::cout << integer << ',';
				}
				std::cout << '}';
			}
			std::cout << ':' << mid << std::endl;
		}
		//std::cout.unsetf(std::ios::hex);
	}

	struct Money {
		int m_type;
		int m_value = 0;
	};

	Money* countMoney() {
		int input;
		const size_t size = 6;
		//int types[]{ 100, 50, 20, 10, 5, 1 };
		Money* moneys = new Money[size]{ {100,0},{50,0},{20,0},{10,0},{5,0},{1,0} };
		//Money* moneys = new Money[size];
		std::cout << "Please input your money:" << std::endl;
		std::cin >> input;

		for (int i = 0; i < size; i++) {
			moneys[i].m_value = input / moneys[i].m_type;
			//moneys[i] = { types[i] ,input / types[i] };
			input %= moneys[i].m_type;
		}
		return moneys;
	}

	void initCountMoney() {
		Money* moneys = countMoney();
		for (int i = 0; i < 6; i++) {
			Money m = moneys[i];
			std::cout << m.m_type << "元的张数是：" << m.m_value << std::endl;
		}
	}

	void testOutput() {
		std::cout.setf(std::ios::hex, std::ios::basefield);
		std::cout << 100 << " ";
		std::cout.unsetf(std::ios::hex);
		std::cout << 100 << " ";
	}

	// get the longest palindrome, input `abase`, return `aba`
	std::string longestPalindrome(std::string s) {
		std::string palindrome = "";
		int targetIndex = 0;
		int longest = 0;
		for (int i = 0, l = s.size();i < l;i++) {
			if ((l - i) * 2 - 1 <= longest) {
				break;
			}
			int prevIndex = i, nextIndex = i;
			bool isOdd = true, isEven = true;
			while (isOdd || isEven) {
				if (isOdd) {
					if (prevIndex < 0 || nextIndex >= l || s[prevIndex] != s[nextIndex]) {
						isOdd = false;
						int length = nextIndex - prevIndex - 1;
						if (length > longest) {
							targetIndex = i;
							longest = length;
						}
					}
				}
				if (isEven) {
					int evenNextIndex = nextIndex + 1;
					if (prevIndex < 0 || evenNextIndex >= l || s[prevIndex] != s[evenNextIndex]) {
						isEven = false;
						int length = evenNextIndex - prevIndex - 1;
						if (length > longest) {
							targetIndex = i;
							longest = length;
						}
					}
				}
				prevIndex--;
				nextIndex++;
			}
		}
		palindrome = s.substr(targetIndex - longest / 2 + (longest % 2 == 0 ? 1 : 0), longest);
		return palindrome;
	}

	//	try get the best team with highest score sum, the only rule is not allow a younger teamate have a higher score than the older one.
	int bestTeamScore(std::vector<int>& scores, std::vector<int>& ages) {
		int sum = 0;
		int size = scores.size();
		std::vector<std::pair<int, int>> interview(size);
		std::vector<int> dp(size, 0);
		for (int i = 0; i < size; i++) {
			interview[i] = { scores[i], ages[i]};
		}
		std::sort(interview.begin(), interview.end());
		for (int i = 0; i < size; i++) {
			int j = i - 1;
			while (j >= 0) {
				if (interview[i].second <= interview[j].second) {
					dp[i] = max(dp[i], dp[j]);
				}
				j--;
			}
			dp[i] += interview[i].first;
			sum = max(sum, dp[i]);
		}
		return sum;
	}

/*
0        8          16
1     7  9       15 17
2   6   10    14    18
3 5     11 13       19
4       12          20

0   4    8	  12
1 3 5 7  9 11 13
2   6   10
=>
0	 1	  2	   3
4 5  6 7  8 9 10
11  12	 13

P     I     N
A   L S   I G
Y A   H R   
P     I

*/
	std::string convert(std::string s, int numRows) {
		int size = s.size();
		if (numRows == 1 || numRows >= size || size <= 2)
			return s;
		int loopSize = (numRows - 1) * 2;
		int c = (size + loopSize - 1) / loopSize * (numRows - 1);
		std::vector<std::string> mat(numRows, std::string(c, 0));
		for (int i = 0, x = 0, y = 0; i < size; ++i) {
			mat[x][y] = s[i];
			if (i % loopSize < numRows - 1)
				++x;
			else {
				--x;
				++y;
			}
		}
		std::string result;
		for (std::string& row : mat) {
			for (char c : row) {
				if (c)
					result += c;
			}
		}
		return result;
	}
}