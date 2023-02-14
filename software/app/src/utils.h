#pragma once
//	std::vector����Ҫ����
#include <vector>
//	std::function���裬���ڶ���ص������ú���
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
		//	���캯����ʵ����ʱ���õķ��������ƺ�����һ������Ҫ��ʼ�����е�ʵ������
		//	explicit�ؼ��ֽ�ֹ����ת������Player player = PlayerLevel_EntryLevel;
		explicit Player(PlayerLevel level);
		//	�ݻ���ʵ��ʱ���õķ���������Ϊ��~������
		~Player();
		void move(int new_x, int new_y);
	};
	//	���ز�����
	class Vec {
	public:
		float m_x, m_y;
		Vec(float x, float y);
		Vec add(const Vec& _vec) const;
		//	���ز������Ӻ�
		Vec operator+(const Vec& _vec) const;
		Vec multiply(const Vec& _vec) const;
		//	���ز������˺�
		Vec operator*(const Vec& _vec) const;
		bool isEqual(const Vec& _vec) const;
		//	������Ȳ�����
		bool operator==(const Vec& _vec) const;
	};

	class Vecv {
	public:
		//	ʹ�û����ų�ʼ�����õ���
		Vec vec{ 2.0f,2.0f };
		Vec& getVec();
	};
	void fastMove(Player& player, int new_x, int new_y);
	//	struct ��Ϊ�˼���c�﷨����class������ֻ��struct�ڵ�ֵĬ����public����classĬ�϶���private
	struct NormalPerson {
		int m_positionX, m_positionY;
		int m_speed;
		Player* m_like;
		void move(int new_x, int new_y);
		void follow(Player& _player);
	};
	//	�������η�
	class Trainer {
		//	ֻ�ܴ������
	private:
		int m_runLevel;
		int m_runNumber;
		//	�ɱ�����ͼ̳������
	protected:
		int m_age;
		int m_sex;
		//	���д��붼�ɵ���
	public:
		Trainer(int runNumber, int age, int sex);
	};

	//	ͨ���麯��ʵ�ֳ�����/�ӿ�
	class Runner {
	public:
		virtual void run() = 0;
	};

	//	�̳�
	class Racer : public Runner {
	public:
		char m_cup;
		int m_rank;
		//	��ʼ���б���ʽ�Ĺ��캯��
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
		//	ʹ��const��β�����������Ϊ�����޸���
		const int getX() const;
	};

	void initConst();

	void initLabbda();

	void initCalculate();

	int* createArray();

	//	����ջ�����ݻٶ���
	class Entity {
	public:
		void dododo();
	};

	class ScopeEntity {
	private:
		Entity* m_entity;
	public:
		//	������ϵ�ʵ��
		ScopeEntity(Entity* entity);
		//	ɾ�����ϵ�ʵ��
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
		//	����ʱ����õĹ��캯��
		SS(const SS& ss);
		~SS();
		void print() const {
			std::cout << m_buffer << std::endl;
		}
		char& operator[](unsigned int index) {
			return m_buffer[index];
		}
		//	��Ԫ��������������˽�б���Ҳ���ⲿ��������
		friend void fri(SS& ss, const char* content);
	};

	//	��Ԫ�������壬���Ե�����������ʵ��˽������
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

	//	���ڶ෵��ֵ��struct
	struct Return1 {
		std::string x;
		std::string y;
		int z;
	};

	//	����struct�෵��ֵ
	Return1 returnStruct();

	//	���ڴ��ݶ�����ò������෵��ֵ
	void returnParams(std::string& str1, std::string& str2, int& z);

	//	��������෵��ֵ
	std::array<std::string, 2> returnArray();

	//	�����Զ���Ķ෵��ֵ
	std::tuple<std::string, std::string, int> returnTuple();

	//	�෵��ֵ������1��struct��2���������ò����ٸ�ֵ��3���������飻4��tuple��������ͬ����ֵ��
	void initReturn();

	//	template����ͨ��ָ��������������ν�ĺ������ض���
	template<typename FirstParam>
	void template1(FirstParam param);

	//	template��������ı������ͺ������С
	template<typename Arr, int size>
	class SArray {
	private:
		Arr arr[size];
	public:
		int getSize() const;
	};

	void initTemplate();

	template<typename Value>
	//	����β��ﶨ��Ļص��������������ͻᵼ��lambda�޷�ʹ��[]����������������ᱨ�����������
	//void each(const std::vector<Value>& values, void(*handler)(Value));
	//	�β����ñ�׼�ⷽ��ģ�嶨��ص��������ͣ�lambda����ʹ��[]�������������
	void each(const std::vector<Value>& values, const std::function<void(Value)>& handler);

	void initLambda();

	void initAuto();
}