#pragma once
#include "./include/stdafx.h"

namespace utils {
	void useLibrary();
	void variableAndLog();
	void incrementWithPointer(int*);
	void incrementWithReference(int&);
	void PointerAndReference();
	void localStaticVar();
	void initStatic();
	enum PlayerLevel;
	enum  class PlayerStatus;
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
		virtual ~Player();
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
		//	���캯����������ϵ�ʵ��Entity
		ScopeEntity(Entity* entity): m_entity(entity) {
		};
		//	����������ɾ�����ϵ�ʵ��Entity
		virtual ~ScopeEntity() {
			delete m_entity;
		}
		//	���ƹ��캯�������ڴ�����ʱ���������Ҫ���´���һ����ʵ��Entity
		ScopeEntity(const ScopeEntity& other): ScopeEntity(other.m_entity) {
		}
		//	�ƶ����캯��
		ScopeEntity(ScopeEntity&& other) noexcept: m_entity(std::exchange(other.m_entity, nullptr)) {
		}
		//	���Ƹ�ֵ����ֵʱ��Ҫ����ԭ��ַ�������Ͷ��µ�ַ���ƹ���
		ScopeEntity& operator=(const ScopeEntity& other) {
			return *this = ScopeEntity(other);
		}
		ScopeEntity& operator=(ScopeEntity&& other) noexcept {
			std::swap(m_entity, other.m_entity);
			return *this;
		}
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
		virtual ~SS();
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
		float x, y;
		Vex(float, float);
	};

	template<typename Vec>
	void print_vector(const std::vector<Vec>&);

	void initVector();

	//	ͨ�����������ù����ڴ��ַ��˫�����ͣ����������4���������Ϳ����滻��2��Vex���ͣ���ΪVex��2���������ͣ�
	union Vex4 {
		struct {
			float p1, p2, p3, p4;
		};
		struct {
			Vex a, b;
		};
	};

	void initUnion();

	//	���ͺ�˫�ؾ��ȸ�����Ҳ��˫�����ͣ�������Ϊ������
	union NumberValue {
		int nvi;
		double nvd;
	};
	//	ʵ���л�ʹ������xyz�ռ�����ͬʱ����rgb��ɫʹ��

	//	���ڶ෵��ֵ��struct
	struct Return1 {
		std::string x;
		std::string y;
		int z;
	};

	//	����struct�෵��ֵ
	Return1 returnStruct();

	//	���ڴ��ݶ�����ò������෵��ֵ
	void returnParams(std::string&, std::string&, int&);

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

	void initAuto();

	void initThread();

	void initSort();

	void initGetFile();

	namespace Hazel {
		void initLockAndAsync();
	}

	void initStringOptimization();

	bool isPalindrome(int);

	void initListNumberAdd();

	int lengthOfLongestSubstring(std::string);

	std::string checkPassword(std::string);

	void testCheckPassword();

	struct Tree_node {
		int val;
		std::vector<Tree_node*> children;
		Tree_node(int val)
			: val(val) {}
		Tree_node(int val, std::vector<Tree_node*>& children)
			: val(val), children(children) {}
	};

	std::vector<int> tree_node_to_array(Tree_node*);

	void test_tree_node2array();

	void testQuickSort();

	double findMedianSortedArrays(std::vector<int>&, std::vector<int>&);

	void initFindMedianSortedArrays();

	void initCountMoney();

	std::string longestPalindrome(std::string);

	std::string convert(std::string, int);

	int strongPasswordChecker(std::string);

	void testStrongPasswordChecker();

	std::string reverseParentheses(std::string);

	void testReverseParentheses();

	void initLambda();

	void reverse_num();

	void test_reverse_num();

	int reverse_number_position();

	void test_reverse_number_position();

	int atoi(std::string);

	void test_atoi();

	int count_time(std::string);

	void test_count_time();

	std::vector<int> add_negabinary(std::vector<int>&, std::vector<int>&);

	void test_add_negabinary();

	bool regex_match(const std::string&, const std::string&);

	void test_regex_match();
}
