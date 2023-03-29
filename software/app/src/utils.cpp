#pragma once
#include "./include/stdafx.h"
//	std::mutex ������
#include <mutex>
//	std::async �첽����
#include <future>
//	��̬�⣬��Ҫ�����exe�ļ��У�Ч�ʸ��ã������ļ�����
//	��̬�⣬һ���Ƿ��õ�exe�ļ��Ա�
//	����������
//#include <GLFW/glfw3.h>
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
		//int a = glfwInit();
		//std::cout << a << std::endl;
		engine::printMessage();
	}

	void variableAndLog() {
		logger::out("hello world");
		const char* name = "c++";
		std::string s = "This's string";
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

	//	��ͳö�٣�ö����ֻ����Ϊ���Ͷ�������ռ���ڣ����Զ����ڲ�����������Ϊ��ǰ���������ʹ��
	enum PlayerLevel {
		PlayerLevel_EntryLevel,
		PlayerLevel_Intermediate,
		PlayerLevel_Expert
	};

	//	ö���࣬ö����ͬʱ��Ϊ���ͺ������ռ���ڣ�����ǰ׺�����������ܵ���
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
			iarr[i] = i;
		}
		std::cout << *iarr << std::endl;
		//	��Ϊ�ڶ��Ϸ������ڴ棬ʹ������Ҫɾ��
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
	//  �ִ�C++������������ʹ������ָ�����ԭʼָ�룬����ֻ��ѧϰ�ͻ��۾�����ʹ��ԭʼָ�룬�����Լ���������ָ�롣

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

	Vex::Vex(float _x, float _y) : x(_x), y(_y) {
	};

	void initUnion() {
		using namespace std;
		Vex4 v4 = { 1.0f, 2.0f, 3.0f, 4.0f };
		//	���Ե���2��Vex���������Ҳ���Ե���4�������������������ҹ����ڴ�
		cout << v4.a.x << ',' << v4.a.y << ',' << v4.b.x << ',' << v4.b.y << endl;
		cout << v4.p1 << ',' << v4.p2 << ',' << v4.p3 << ',' << v4.p4 << endl;
	}

	//	����������������Vex��
	std::ostream& operator<<(std::ostream& stream, const Vex& vex) {
		stream << vex.x << ',' << vex.y;
		return stream;
	}

	template<typename Vec>
	void outputVex(const std::vector<Vec>& vexs) {
		//	ѭ����ȡ��
		for (const Vec& vex : vexs) {
			std::cout << vex << std::endl;
		}
	}

	//	��̬����
	void initVector() {
		//	������������������̬����
		std::vector<Vex> vexs;
		//	��ʱ���������ݣ�������Զ����ݳ�������ȥ���´��������飨��ɾ�������飩
		//	push_back���ڵ�ǰ��������һ��Vexʵ����֮��Ÿ��ƽ�vector��
		vexs.push_back({ 0.0f, 0.0f });
		vexs.push_back({ 1.0f, 4.0f });
		outputVex<Vex>(vexs);
		//	�����������
		vexs.clear();
		//	emplace_back��ֱ����vector�ڲ��������Ͳ������ȴ����ٸ��Ƶ��µ�Ч������
		vexs.emplace_back(1.0f, 1.0f);
		vexs.emplace_back(2.0f, 7.0f);
		outputVex<Vex>(vexs);
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
		logger::out("�߳̽���");
	}

	void initSort() {
		std::vector<int> vecInt = { 1, 5, 7, 3 , 2 };
		//  �������򷽷���Ĭ�ϰ��մ�С�����������û�д����жϷ����Ļ�
		std::sort(vecInt.begin(), vecInt.end(), [](int a, int b) {
			return a < b;
			});
		utils::outputVex<int>(vecInt);
	}

	//	����ת��������RTTI(Runing Time Type Info������ʱ������Ϣ�������ø����ú�ʹ��C++�����������תת
	void initTypeConvertionCheck() {
		double a = 5.25;
		int b1 = a;	//	��ʽת����C���
		int b2 = (double)a;	//	��ʽת����C���
		//	������C++��������ת����ʵ������һ���������ж��⿪����
		//	��̬����ת��������������ڱ�������н׶�ȷ������Ŷ�����������ں���ά�������������������ת��
		int b = static_cast<int>(a);
		//	����˫�ص�ת��
		logger::timer timer("initTypeConvertionCheck repinterpret_cast");
		double* t = (double*)reinterpret_cast<logger::timer*>(&timer);
		//	��̬����ת����������ȷ�ϼ̳й�ϵ�µ�����
		Racer* racer = new Racer(*"worker", 1);
		Runner* runner = racer;
		Racer* newRacer = dynamic_cast<Racer*>(runner);
		//	������ָ�븳ֵ������ָ������󣬿�ͨ����̬����ת��ȷ���Ƿ�Ϊĳ�ض��������ͣ�ת��������ֵ���Ǹ��������ͣ�ʧ��Null����
		if (newRacer) {

		}

		//	�����������ת��
		const char* cc = "hello";
		char* dd = const_cast<char*>(cc);
	}

	//	��ȡ���ܲ����ڵ��ļ���������Ϊ��ѡ����ֵ
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

	//	ָ�������͵ĵ�һ����variant�����union��ע�����Ͱ�ȫ����union����Ч�ʺ͸����ڴ�ռ��
	void sigleVariantMultiType() {
		std::variant<std::string, int> data;
		//	�������ַ���
		data = "lotaway";
		//	��Ҫָ����ȡ������
		logger::out(std::get<std::string>(data));
		//	Ҳ��������ֵ
		data = 30;
		logger::out(std::get<int>(data));
		//	�޷�ȷ���������͵�����£������ͨ���жϻ�ȡ
		logger::out(*std::get_if<std::string>(&data));
	}

	//	std::any �������͵ĵ�һ���������Ƽ��ã������variant���������������п��ܵ����ͣ���ͬ�ĵ�����ȡֵʱ��Ҫָ�����ͣ�ȱ������Ҫ��̬�����ڴ浼����������
	void anyValue() {
		std::any data;
		data = "lotaway";
		data = 30;
		logger::out(std::any_cast<int>(data));
	}

	//	ͨ���첽�����̣߳����д���������������

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
		//	������
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

	//	�ַ����Ż�������Ǽ���ʹ��string����char�����ַ�����string_view
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
	//	ʵ����Release��������ʱ����SSOС�ַ����Ż������Զ����ַ�������ַ�����ջ���������Ƕ�ȥ�����ڴ棬ֻ�бȽϳ����ַ��Ż������öѷ����ڴ棬��VS2019�У��������ֻ��Ƶĳ�����15���ַ���

	//	���ģʽ������ģʽ
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

	//	��ֵ����ֵ���о���λ�õ�����ֵ��ֻ����ʱֵ������ֵ����ֵû��λ�ã����Բ��ܱ���ֵ
	int getValue() {
		return 10;
	}
	void setValue(int val) {
	}
	//	����ʱ��ֻ�ܴ�����ֵ�����ܴ�����ֵ
	void setValue(std::string& name) {

	}
	//	��������ʱ�����Դ�����ֵ����ֵ
	void setValue2(const std::string& name) {

	}
	//	˫������ʱ��ֻ�ܴ�����ֵ�����ܴ�����ֵ
	void setValue3(std::string&& name) {

	}
	void initLValueAndRValue() {
		//	����a����ֵ��1����ֵ
		int a = 1;
		//	����a��b������ֵ
		int b = a;
		//	����c����ֵ��getValue()����һ����ֵ
		int c = getValue();
		//	���ܶ�getValue()��ֵ����Ϊ������һ����ֵ
		//getValue() = a;
		//	����a����ֵ
		setValue(a);
		//	����2����ֵ
		setValue(2);
		
		//	firstName����ֵ��"Way"����ֵ
		std::string firstName = "Way";
		//	lastName����ֵ��"Luk"����ֵ
		std::string lastName = "Luk";
		//	�����ұ���������ֵfirstName��lastName����firstName + lastName������������һ����ʱֵ����������ֵ
		std::string fullName = firstName + lastName;

		//	����ʱ��ֻ�ܴ�����ֵ
		setValue(fullName);
		//	����ʱ�����ܴ�����ֵ���˴��ǳ�����������ֵ
		//setValue("lotaway");
		// 
		//	��������ʱ�����ܴ�����ֵ
		setValue2(fullName);
		//	��������ʱ��Ҳ�ܴ�����ֵ
		setValue2("Way Luk");

		//	˫������ʱ��ֻ�ܴ�����ֵ
		setValue3("Way Luk");
		//	˫������ʱ�����ܴ�����ֵ
		//serValue3(fullName);

		//	�����������ִ�������д���ط���������ƶ����壬�����ݵ�����ֵʱ�����Է��Ľ���ʹ�������޸ģ���Ϊֻ����ʱʹ�ö����Ḵ�ƻ���Ӱ���������ݡ�
	}

	//	�ƶ�����
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
			//	��˫�����������õ���ֵ�����򵥵��ƶ�ָ�룬֮��ɾ��ԭָ��
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
		//	�β���name����ֵ�����ŵ���ʼ���б����ʵ��name����Ϊ��ֵ���룬��Ҫʹ��std;:move�������ٴ�ǿ�Ʊ����ֵ
		Enstring(String&& name): m_name(std::move(name)) {}
	private:
		String m_name;
	};

	void initStringAndMove() {
		//	�����ʹ��˫������&&����String���췽�������������ʽ�����ڶ��Ϸ��䴴��String("lotaway")��֮����Enstringͨ��m_name��ʼ���б��������ڶ��Ϸ��䴴������ʹʹ�õ�������&����String�Ĺ��촴���븴�Ʒ��������˻����������ѣ�
		Enstring enstring("lotaway");
	}

	// iteralor��������ʹ��ָ�����ѭ������ȡֵ
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
		//	method 3: using for : with �ṹ�⹹
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

	//	˫�ؼ������

	// hash map �ٷ�ʵ�֣�unordered_map�������Ҽ���Ψһ�����Ͱ���䷽ʽ
	void initHashMap() {
		//	��Ҫ�ṩ�������ͺͼ�ֵ���Ͳ��ܴ���hashmap
		std::unordered_map<std::string, int> umap;

		//	������ݣ���ԭ���ļ��Ѿ������򸲸ǣ�
		umap.emplace("wayluk", 30);
		umap.emplace("lotaway", 18);

		//	��ȡ����
		std::unordered_map<std::string, int>::const_iterator it = umap.find("wayluk");
		if (it == umap.end())
			std::cout << "�Ҳ�������" << std::endl;
		else
			std::cout << "�ҵ��ˣ�����" << it->second << std::endl;

		//	ɾ�����ݣ��˴�Ϊ����ɾ����Ҳ���Ե�������Χ��ɾ��
		umap.erase("wayluk");

		//	���������ݣ�ֻ������¼����Ḳ�����У�����������������ֵ������map��
		//	��ʽ1������pair
		std::pair<std::string, int> newGuy("wayluk", 31);
		umap.insert(newGuy);
		//	��ʽ2��������һ��map��һ���ֻ���ȫ����
		std::unordered_map<std::string, int> other_map{ { "mimi", 27 }, {"haha", 37} };
		umap.insert(other_map.begin(), other_map.end());
		//	��ʽ3��ֱ����������ʼ��
		umap.insert({ { "shutup", 30 }, { "hate", 30 }, {"eatshit", 30} });

		//	��������Χ��ɾ��
		if (!umap.empty())
			umap.erase(other_map.begin(), other_map.end());

		//	ѭ�����
		for (auto& m : umap)
			std::cout << m.first << ":" << m.second << std::endl;
	}

	//	��hashmap������������ҵ�ָ�������������ӣ����ṩ����[30,40,60,80]����ֵ100������40+60=100��Ҫ���ҵ�40��60���������ǵ�����ֵ
	std::vector<int> getSumTwoBySum(const std::vector<int>& arr, const int sum) {
		std::unordered_map<int, int> requireUMap;
		for (int i = 0, l = arr.size(); i < l; i++) {
			//	���ҵ����������ֵһ��ʱ�������ǰ�����ֵ������map���������ڵ���ֵ����Ϊһ�ԣ��õ������
			std::unordered_map<int, int>::const_iterator it = requireUMap.find(sum - arr[i]);
			if (it != requireUMap.end())
				return { it->second, i };
			//	����ǰ����Ҫ����ֵ������ֵ��¼����
			requireUMap.emplace(arr[i], i);
		}
		return { 0, 0 };
	}

	//	�ж�����������Ƿ�Ϊ���ĸ�ʽ����121��12321���ǻ��ģ��Ӹ�λ����λ������Ҳ����ȣ�ע�ⲻҪʹ���ַ�����ʽ
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

		//	��ת�ַ�������ʽ1����Ҫ��������һ���ַ���
		std::string reStr(originStr.rbegin(), originStr.rend());

		//	��ת�ַ�������ʽ2��ֱ���޸�ԭ�е��ַ���
		std::reverse(originStr.begin(), originStr.end());

		//	��ת�ַ�������ʽ3������һ���ַ������з�ת
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

	//	��ȡ�ַ�����÷û���ظ��ַ������ַ������ȣ���abcacd�У�abc��������ظ��ַ������ַ���������Ϊ3
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
			std::cout << m.m_type << "Ԫ�������ǣ�" << m.m_value << std::endl;
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