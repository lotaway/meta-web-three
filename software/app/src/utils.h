#pragma once
//	std::vector类需要引入
#include <vector>
//	std::function所需，用于定义回调和引用函数
#include <functional>

namespace utils {
	void useLibrary();
	void variableAndLog();
	void incrementWithPointer(int* value);
	void incrementWithReference(int& value);
	void PointerAndReference();
	void localStaticVar();
	void initStatic();
	enum PlayerLevel;
	enum PlayerStatus;
	class Player {
	private:
		PlayerLevel m_level;
		PlayerStatus m_status;
	public:
		int m_positionX, m_positionY;
		int m_speed;
		//	构造函数，实例化时调用的方法，名称和类名一样，需要初始化所有的实例变量
		//	explicit关键字禁止隐性转换，如Player player = PlayerLevel_EntryLevel;
		explicit Player(PlayerLevel level);
		//	摧毁类实例时调用的方法，名称为【~类名】
		~Player();
		void move(int new_x, int new_y);
	};
	//	重载操作符
	class Vec {
	public:
		float m_x, m_y;
		Vec(float x, float y);
		Vec add(const Vec& _vec) const;
		//	重载操作符加号
		Vec operator+(const Vec& _vec) const;
		Vec multiply(const Vec& _vec) const;
		//	重载操作符乘号
		Vec operator*(const Vec& _vec) const;
		bool isEqual(const Vec& _vec) const;
		//	重载相等操作符
		bool operator==(const Vec& _vec) const;
	};

	class Vecv {
	public:
		//	使用花括号初始化引用的类
		Vec vec{ 2.0f,2.0f };
		Vec& getVec();
	};
	void fastMove(Player& player, int new_x, int new_y);
	//	struct 是为了兼容c语法，与class的区别只有struct内的值默认是public，而class默认都是private
	struct NormalPerson {
		int m_positionX, m_positionY;
		int m_speed;
		Player* m_like;
		void move(int new_x, int new_y);
		void follow(Player& _player);
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
		Trainer(int runNumber, int age, int sex);
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
		Racer(const char& cup, int rank);
		void run() override;
	};

	class Winner : public Racer {
	public:
		std::string getNews();
	};

	class InitStatic {
	public:
		static const int s_defaultSpeed = 2;
		static int s_maxSpeed;
	};

	void initClass();

	void initArray();

	void PrintString(const std::string& str);

	void initString();

	class OnlyReadFn {
	private:
		int m_x;
		mutable int getCount;
	public:
		OnlyReadFn();
		//	使用const在尾部将函数标记为不会修改类
		const int getX() const;
	};

	void initConst();

	void initLabbda();

	void initCalculate();

	int* createArray();

	//	利用栈类来摧毁堆类
	class Entity {
	public:
		void dododo();
	};

	class ScopeEntity {
	private:
		Entity* m_entity;
	public:
		//	传入堆上的实例
		ScopeEntity(Entity* entity);
		//	删除堆上的实例
		~ScopeEntity();
	};

	void initStackClass();

	void initIntelligencePointer();

	class SS {
	private:
		char* m_buffer;
		unsigned int m_size;
	public:
		SS(const char* content);
		//	拷贝时会调用的构造函数
		SS(const SS& ss);
		~SS();
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
	void fri(SS& ss, const char* content);

	void stringCopy();

	class Origin {
	public:
		void print() const;
	};

	class SpecPointer {
	private:
		Origin* origin;
	public:
		SpecPointer(Origin* _origin);
		const Origin* operator->() const;
	};

	void arrowPoint();

	struct Vex {
		float x, y, z;
		Vex(float _x, float _y, float _z);
	};

	void outputVex(const std::vector<Vex>& vexs);

	void initVector();

	//	用于多返回值的struct
	struct Return1 {
		std::string x;
		std::string y;
		int z;
	};

	//	利用struct多返回值
	Return1 returnStruct();

	//	用于传递多个引用参数并多返回值
	void returnParams(std::string& str1, std::string& str2, int& z);

	//	用于数组多返回值
	std::array<std::string, 2> returnArray();

	//	返回自定义的多返回值
	std::tuple<std::string, std::string, int> returnTuple();

	//	多返回值方法：1、struct；2、传递引用参数再赋值；3、返回数组；4、tuple定义多个不同类型值。
	void initReturn();

	//	template可以通过指定泛型来减少无谓的函数重载定义
	template<typename FirstParam>
	void template1(FirstParam param);

	//	template定义类里的变量类型和数组大小
	template<typename Arr, int size>
	class SArray {
	private:
		Arr arr[size];
	public:
		int getSize() const;
	};

	void initTemplate();

	template<typename Value>
	//	如果形参里定义的回调函数是匿名类型会导致lambda无法使用[]捕获作用域变量，会报错参数不符合
	//void each(const std::vector<Value>& values, void(*handler)(Value));
	//	形参里用标准库方法模板定义回调函数类型，lambda才能使用[]捕获作用域变量
	void each(const std::vector<Value>& values, const std::function<void(Value)>& handler);

	void initLambda();

	void initAuto();
}