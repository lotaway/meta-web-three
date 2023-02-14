#pragma once
#include <iostream>
//	std:array ��Ҫ
#include <array>
//	cout << std::to_string ʱ��Ҫ������intתΪstring
#include <string>
//	unique_ptr����ָ������Ҫ����
#include <memory>
#include <vector>
//	make_tuple��Ҫ
#include <tuple>
//	��̬�⣬��Ҫ�����exe�ļ��У�Ч�ʸ��ã������ļ�����
//	��̬�⣬һ���Ƿ��õ�exe�ļ��Ա�
//	����������
#include <GLFW/glfw3.h>
//	�����������е�������Ŀ
// emsdk�޷�ʶ��ֻ��ʹ�����ż����·��"../../engine/src/engine.h"������cpp�ͱ�׼��������ļ���û�б������ȥwasm
#include <engine.h>
#include "logger.h"
#include "utils.h"

#define debugger(msg) std::cout << "main::debugger:" + msg << std::endl

extern int g_variable;	//	�����ⲿ�����ļ����ҵ����ȫ��ֵ��Ŀǰ������log.cpp��

namespace utils {
	//	��Դ�ھ�̬���̬��ķ���
	void useLibrary() {
		int a = glfwInit();
		std::cout << a << std::endl;
		engine::printMessage();
	}

	void variableAndLog() {
		logger::out("hello world");
		const char* name = "c++";
		std::string s = "This's string~~";
		//std::wstring s = "������ַ�����";
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
		int mVal = logger::multiply(2, 3);
		logger::out("mVal", mVal);
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
		//*ptr = 10;	//	����voidָ���޷�����10��ʲô��������д��
		int* ptr1 = &var;
		*ptr1 = 10;	//	��ȷ��intָ������10Ӧ�����������ͣ�����д��
		double* ptr2 = (double*)&var;
		//log((char*)&ptr2);
		std::cin.get();

		char* buffer = new char[8];	//	����8�ֽ��ڴ�
		memset(buffer, 0, 8);	//	����������ڴ�
		//	ִ��һЩ����
		char** ptr3 = &buffer;
		//	ִ��һЩ����֮�󡣡���
		delete[] buffer;	//	ɾ��ָ�룬��Ϊ������ڴ�
		std::cin.get();
	}

	//	ǿ�󣺴���ָ������޸�
	void incrementWithPointer(int* value) {
		(*value)++;
	}

	//	�򻯣��������ý����޸ģ��Ƽ�ʹ�����ַ�ʽ
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
		//	ref�Ƕ�a�����ã���c���ֵ��ֵ��ref��a����a=8��ref=8������c=8�����߻�������
		ref = c;
		logger::out("ref", ref);
		//	ָ����Ա��޸�ָ���µĵ�ַ�������ò����ԣ�ֻ�ܸ�ֵ
		b = &c;
		c = 10;
		logger::out("b change", *b);
	}

	void localStaticVar() {
		//	�ֲ���̬����ʵ�����ǳ������ڵľ�̬��������һ�ε��û��ʼ����ֵΪ1����֮�󱣳�������2��3��4...��
		static int variable = 1;
		variable++;
		std::cout << "local static variable " + std::to_string(variable) << std::endl;
	}

	void initStatic() {
		logger::out("extern global variable", g_variable);
		//	���ö�κ����ڽ������ͬ�ľֲ���̬����
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

	Player::Player(PlayerLevel level = PlayerLevel_EntryLevel) : m_level(level), m_status(PlayerStatus_Enabled), m_positionX(0), m_positionY(0), m_speed(15) {

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

	//	��ʼ����̬����ֻ���������ȫ�ַ�ʽ���
	//	��ʽ1��ͨ����������
	int InitStatic::s_maxSpeed = 3;
	//	��ʽ2����ʵ����Ҳ�������ó�Ա����һ�����þ�̬����
	//InitStatic initState;
	//int initState.s_maxSpeed = 3

	void initClass() {
		//	��ջ��ʵ������ֻ���������ڴ��ڣ��뿪��������Զ��ͷ��ڴ棬��ջֻ��1~2MB��С
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
		//	p1ָ��Ϊ�յ�Player�࣬��Ϊԭ����player�Ѿ�����
		std::cout << p1 << std::endl;
		//	�ڶ���ʵ����������������������ĵط����ڣ���ʹ��ʱ��Ҫ�ֶ��ͷ��ڴ�
		Player* p2;
		{
			Player* player2 = new Player();
			player2->m_positionX = 1;
			p2 = player2;
		}
		//	*p2ָ���player2���ɴ���
		std::cout << p2 << std::endl;

		// ͨ������ʽֱ�ӳ�ʼ����Ա������ֻ������û���Զ��幹�캯�������
		NormalPerson normalPerson = { 0, 0, 10 };

		//	new���ų�ʼ��ʱֻ�᷵��ָ�룬��Ҫ��*�洢
		Player* player3 = new Player();
		player3->m_positionX = 0;

		NormalPerson* normalPerson2 = new NormalPerson();
		//	������ָ�룬����ͨ��ʹ���Ǻŵ��û����ó�Ա����
		(*normalPerson2).m_speed = 4;
		//	Ҳ����ͨ����ͷ->���û����ó�Ա����
		normalPerson2->m_speed = 5;
		//	������Ƕ����ĺ���������������main�з������ڴ�ռ䣬��ʹ����Ҫ����ɾ��
		delete normalPerson2;

		std::cin.get();

	}

	void initArray() {
		const int arrSize = 5;
		//	���嶨���������飬������ͨ��ѭ�����г�ʼ��
		int arr[arrSize];
		std::cout << arr << std::endl;
		//	ʵ�������壬ʵ��������ָ�룬����ͨ��ѭ�����г�ʼ��
		int* iarr = new int[arrSize];
		//	������ʹ�ã���ȡ�������鳤�ȣ�û�п���ֱ�ӻ�ȡ���ȵ����Ի򷽷���sizeof��ȡ�����ֽڳ��ȣ�
		//int arrSize = sizeof(iarr) / sizeof(int);
		for (int i = 0;i < arrSize;i++) {
			iarr[i] = 0;
		}
		//	��Ϊ�������ڴ棬ʹ������Ҫɾ��
		delete[] iarr;
		//	��ʽ����
		std::array<int, arrSize> stdArray;
		std::cout << stdArray[0] << std::endl;
		delete& stdArray;
	}

	//	ͨ�����������ַ���ʱ���ͨ�����ô��ݣ�������ɸ��ƣ��������ڴ棩
	void PrintString(const std::string& str) {
		std::cout << str << std::endl;
	}

	void initString() {
		//	�ַ�ָ�룬ʵ�������飬�ɵ����ַ���ʹ�ã�ʵ���ַ��Ƿֿ��洢Ϊ['l','o','t','a','w','a','y']
		const char* cname = "lotaway";
		//	�ַ����飬utf8��ע�ⳤ����7����ʵ���ַ���1λ����Ϊ�������Զ����\0��Ϊ�����ַ�
		const char aname[7] = "lotawa";
		//	�������ַ�����
		const wchar_t* cname2 = L"lotaway";
		//	utf16��ÿ���ַ�2�ֽ�
		const char16_t* ccname = u"lotaway";
		//	utf32��ÿ���ַ�4�ֽ�
		const char32_t* cccname = U"lotaway";
		//	��ȡ�ַ����鳤��
		std::cout << strlen(cname) << std::endl;
		//	�����ַ�����
		char* cpyName = new char[5];
		//strcpy_s(cpyName, strlen(cpyName - 1), cname);

		//	�ַ���
		std::string sname = "lotaway";
		std::cout << sname << std::endl;
		//	��ȡ�ַ�������
		std::cout << sname.size() << std::endl;
		//	�ַ���ƴ�ӣ��������ڶ���ʱֱ�����
		sname += "'s name!";
		//	�����ڶ���ʱ�������ַ���
		std::string ssname = std::string("lotaway") + "'s name";
		//	�ж��ַ������Ƿ����ĳЩ���ַ���
		bool contains = ssname.find("name") != std::string::npos;

		//	�﷨��ƴ�ӣ���Ҫ�����
		using namespace std::string_literals;
		std::string uname = "lotaway"s + "'s name";
		std::u32string uuname = U"lotaway"s + U"'s name";
		//	�ɻ����ַ�����
		const char* change = R"(line1
		line2
		line3
		line4
)";
	}

	OnlyReadFn::OnlyReadFn() : m_x(0), getCount(0) {

	}

	const int OnlyReadFn::getX() const {
		//	�����ø�ֵ����Ϊ�Ѿ����Ϊconst
		//m_x = 2;
		//	����mutable���ɿ����޸�,mutable>const>variable
		getCount += 1;
		return m_x;
	}

	void initConst() {
		//	������ֻ��
		const int MAX_AGE = 140;
		//	�����ڳ���
		constexpr int MAX_COUNT = 1;
		//	����ָ�룬��ָ������ָ�롣���޸�ָ�룬�������޸�ָ��ָ�������
		const int* a = new int(2);
		std::cout << *a << std::endl;
		//	�������޸�aָ�������
		//*a = 2;
		//	�����޸�ָ�뱾��
		a = &MAX_AGE;
		std::cout << *a << std::endl;
		//	ָ�볣������ָ���ǳ����������޸�ָ�룬�����޸�ָ��ָ�������
		int* const b = new int;
		std::cout << b << std::endl;
		//	�����޸�bָ�뱾��
		*b = MAX_AGE;
		std::cout << b << std::endl;
		//	�������޸�bָ��ָ�������
		//b = &MAX_AGE;
		//	����ָ�볣�����Ȳ������޸�ָ�룬Ҳ�������޸�ָ��ָ�������
		const int* const c = new int(3);
		std::cout << *c << std::endl;
		//*c = MAX_AGE:
		//c = &MAX_AGE;
		OnlyReadFn onlyReadFn;
		std::cout << onlyReadFn.getX() << std::endl;
	}

	void initLabbda() {
		int a = 8;
		//	lambda���ʽ��һ���Ժ�����&�������ã�=����ֵ����
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

	//	�������<<������Ҳ��������
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
		//	���÷�����ɼ���
		Vec mulRes = vec.add(walkSpeed).multiply(powerUp);
		//	���ò�������ɼ���
		Vec mulRes2 = (vec + Vec(walkSpeed)) * powerUp;
		std::cout << (mulRes == mulRes2) << std::endl;
		//	std::ostream << Vec���͵Ĳ����������ˣ��������Vec����
		std::cout << mulRes2 << std::endl;
	};

	int* createArray() {
		//	ջ�Ϸ��䣬���غ�ͻ����ٱ���������ֵû������
		//int arr[50];
		//	���Ϸ��䣬��һֱ����ֱ���ֶ��ݻ�
		int* arr = new int[50];
		return arr;
	}

	void Entity::dododo() {}

	ScopeEntity::ScopeEntity(Entity* entity) : m_entity(entity) {}

	ScopeEntity::~ScopeEntity() {
		delete m_entity;
	}

	void initStackClass() {
		//	�˴����Զ�����ת�����൱��ScopeEntity(new Entity())
		{
			ScopeEntity se = new Entity();
		}
		//	�뿪�������ջ�ϵ�se�Զ����٣�������Ķ��ϵ�entityʵ��Ҳ����Ϊ�����۹���������ɾ��
	}

	//	����ָ�룬�Զ�����new��delete
	void initIntelligencePointer() {
		//	unique_ptr���ڴ���Ψһָ�룬���ɸ���
		{
			//	����Ψһָ�룬�뿪��������Զ�ɾ�����ϵ�ʵ������
			std::unique_ptr<Entity> uniqueEntity(new Entity());
			//	����
			std::unique_ptr<Entity> uniqueEntity2 = std::make_unique<Entity>();
			uniqueEntity->dododo();
		}
		//	shared_ptr���ڴ�������ָ�룬�ɸ��ƣ������ù������Զ�����
		std::shared_ptr<Entity> shareEntity;
		{
			std::shared_ptr<Entity> shareEntity2(new Entity());
			shareEntity = shareEntity2;
		}
		//	���ɿ��Ե��ã���Ϊentity0��������
		std::cout << shareEntity << std::endl;
		//	weak_ptr���ڴ���������ָ�룬���ᱻ����������
		std::weak_ptr<Entity> weakEntity;
		{
			std::shared_ptr<Entity> shareEntity3(new Entity());
			weakEntity = shareEntity3;
		}
		//	�Ѿ���ɾ������Ϊ����������
		std::cout << weakEntity.lock() << std::endl;
		std::cin.get();
	}

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
		//	ǳ��������char* m_bufferֻ�´��ָ�룬���յ������۹������������ڴ�ʱ������Ϊ�޷�ɾ������
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

	Vex::Vex(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {};

	//	����������������Vex��
	std::ostream& operator<<(std::ostream& stream, const Vex& vex) {
		stream << vex.x << ',' << vex.y << ',' << vex.z;
		return stream;
	}

	void outputVex(const std::vector<Vex>& vexs) {
		//	ѭ����ȡ��
		for (const Vex& vex : vexs) {
			std::cout << vex << std::endl;
		}
	}

	//	��̬����
	void initVector() {
		//	������������������̬����
		std::vector<Vex> vexs;
		//	��ʱ���������ݣ�������Զ����ݳ�������ȥ���´��������飨��ɾ�������飩
		//	push_back���ڵ�ǰ��������һ��Vexʵ����֮��Ÿ��ƽ�vector��
		vexs.push_back({ 0.0f, 0.0f, 0.0f });
		vexs.push_back({ 1.0f, 4.0f, 7.0f });
		outputVex(vexs);
		//	�����������
		vexs.clear();
		//	emplace_back��ֱ����vector�ڲ��������Ͳ������ȴ����ٸ��Ƶ��µ�Ч������
		vexs.emplace_back(1.0f, 1.0f, 1.0f);
		vexs.emplace_back(2.0f, 7.0f, 8.0f);
		outputVex(vexs);
		//	ɾ��ָ��������ֵ���޷�ֱ����number���ͣ�.begin()�൱�ڿ�ʼ��0����ɾ������ֵΪ1�ĵڶ���ֵ
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

	//	template����һ��Ԫ��̣������뱾�����ڱ���ʱȷ���������������ڲ�ȷ��
	void initTemplate() {
		template1("hahah");	//	�����Զ���������ֵ�ƶ����ͣ������ֶ�����template1<std::string>("hahah");
		SArray<int, 5> sarray;
		std::cout << sarray.getSize() << std::endl;
	}

	template<typename Value>
	//	����β��ﶨ��Ļص��������������ͻᵼ��lambda�޷�ʹ��[]����������������ᱨ�����������
	//void each(const std::vector<Value>& values, void(*handler)(Value)) {
	//	�β����ñ�׼�ⷽ��ģ�嶨��ص��������ͣ�lambda����ʹ��[]�������������
	void each(const std::vector<Value>& values, const std::function<void(Value)>& handler) {
		for (Value value : values) {
			handler(value);
		}
	}

	void initLambda() {
		const char* name = "extra";
		using Value = int;
		std::vector<Value> vec = { 1, 2, 3 };
		// ����������û�е�ǰ������ı���
		each<Value>(vec, [](Value val) { logger::out("name", val); });
		// ������������Ҫ�е�ǰ����������б���
		each<Value>(vec, [=](Value val) { logger::out(name, val); });
		// ������������Ҫ�е�ǰ�������ĳ������
		each<Value>(vec, [&name](Value val) { logger::out(name, val); });
	}

	void initAuto() {
		std::vector<int> vec = { 1, 2, 3 };
		auto it = std::find_if(vec.begin(), vec.end(), [](int val) { return val > 2; });
		logger::out(*it);
	}
}