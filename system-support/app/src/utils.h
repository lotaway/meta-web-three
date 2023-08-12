#pragma once
#include "./include/stdafx.h"

namespace utils {
	void use_library();
	void variable_and_log();
	void increment_with_pointer(int*);
	void increment_with_reference(int&);
	void pointer_and_reference();
	void local_static_var();
	void init_static();
	enum player_level;
	enum  class player_status;
	class player {
	private:
		player_level m_level;
		player_status m_status;
	public:
		int m_positionX, m_positionY;
		int m_speed;
		//	���캯����ʵ����ʱ���õķ��������ƺ�����һ������Ҫ��ʼ�����е�ʵ������
		//	explicit�ؼ��ֽ�ֹ����ת������Player player = PlayerLevel_EntryLevel;
		explicit player(player_level);
		//	�ݻ���ʵ��ʱ���õķ���������Ϊ��~������
		virtual ~player();
		void move(int, int);
	};
	//	���ز�����
	class vec2 {
	public:
		float m_x, m_y;
		vec2(float, float);
		vec2 add(const vec2&) const;
		//	���ز������Ӻ�
		vec2 operator+(const vec2&) const;
		vec2 multiply(const vec2&) const;
		//	���ز������˺�
		vec2 operator*(const vec2&) const;
		bool is_equal(const vec2&) const;
		//	������Ȳ�����
		bool operator==(const vec2&) const;
	};

	class vec4 {
	public:
		//	ʹ�û����ų�ʼ�����õ���
		vec2 vec{ 2.0f,2.0f };
		vec2& get_vec2();
	};
	void fastMove(player&, int, int);
	//	struct ��Ϊ�˼���c�﷨����class������ֻ��struct�ڵ�ֵĬ����public����classĬ�϶���private
	struct normal_person {
		int m_positionX, m_positionY;
		int m_speed;
		player* m_like;
		void move(int, int);
		void follow(player&);
	};
	//	�������η�
	class trainer {
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
		trainer(int, int, int);
	};

	//	ͨ���麯��ʵ�ֳ�����/�ӿ�
	class runner {
	public:
		virtual void run() = 0;
	};

	//	�̳�
	class racer : public runner {
	public:
		char m_cup;
		int m_rank;
		//	��ʼ���б���ʽ�Ĺ��캯��
		racer(const char&, int);
		void run() override;
	};

	class winner : public racer {
	public:
		std::string get_news();
	};

	class init_static {
	public:
		static const int s_defaultSpeed = 2;
		static int s_maxSpeed;
	};

	void init_class();

	void init_array();

	void print_string(const std::string&);

	void init_string();

	class only_read_fn {
	private:
		int m_x;
		mutable int getCount;
	public:
		only_read_fn();
		//	ʹ��const��β�����������Ϊ�����޸���
		const int get_x() const;
	};

	void init_const();

	void init_calculate();

	int* createArray();

	//	����ջ�����ݻٶ���
	class entity {
	public:
		void dododo();
	};

	class scope_entity {
	private:
		entity* m_entity;
	public:
		//	���캯����������ϵ�ʵ��entity
		scope_entity(entity* e): m_entity(e) {
		};
		//	����������ɾ�����ϵ�ʵ��entity
		virtual ~scope_entity() {
			delete m_entity;
		}
		//	���ƹ��캯�������ڴ�����ʱ���������Ҫ���´���һ����ʵ��entity
		scope_entity(const scope_entity& other): scope_entity(other.m_entity) {
		}
		//	�ƶ����캯��
		scope_entity(scope_entity&& other) noexcept: m_entity(std::exchange(other.m_entity, nullptr)) {
		}
		//	���Ƹ�ֵ����ֵʱ��Ҫ����ԭ��ַ�������Ͷ��µ�ַ���ƹ���
		scope_entity& operator=(const scope_entity& other) {
			return *this = scope_entity(other);
		}
		scope_entity& operator=(scope_entity&& other) noexcept {
			std::swap(m_entity, other.m_entity);
			return *this;
		}
	};

	void init_stack_class();

	void init_intelligence_pointer();

	class ss {
	private:
		char* m_buffer;
		unsigned int m_size;
	public:
		ss(const char*);
		//	����ʱ����õĹ��캯��
		ss(const ss&);
		virtual ~ss();
		void print() const {
			std::cout << m_buffer << std::endl;
		}
		char& operator[](unsigned int index) {
			return m_buffer[index];
		}
		//	��Ԫ��������������˽�б���Ҳ���ⲿ��������
		friend void fri(ss&, const char*);
	};

	//	��Ԫ�������壬���Ե�����������ʵ��˽������
	void fri(ss&, const char*);

	void string_copy();

	class origin {
	public:
		void print() const;
	};

	class spec_pointer {
	private:
		origin* m_origin;
	public:
		spec_pointer(origin*);
		const origin* operator->() const;
	};

	void arrow_point();

	struct vex2 {
		float x, y;
		vex2(float, float);
	};

	template<typename _vector>
	void print_vector(const std::vector<_vector>&);

	void init_vector();

	//	ͨ�����������ù����ڴ��ַ��˫�����ͣ����������4���������Ϳ����滻��2��Vex���ͣ���ΪVex��2���������ͣ�
	union vex4 {
		struct {
			float p1, p2, p3, p4;
		};
		struct {
			vex2 a, b;
		};
	};

	void init_union();

	//	���ͺ�˫�ؾ��ȸ�����Ҳ��˫�����ͣ�������Ϊ������
	union number_value {
		int nvi;
		double nvd;
	};
	//	ʵ���л�ʹ������xyz�ռ�����ͬʱ����rgb��ɫʹ��

	//	���ڶ෵��ֵ��struct
	struct return_struct {
		std::string x;
		std::string y;
		int z;
	};

	//	����struct�෵��ֵ
	return_struct return_mutiply_struct();

	//	���ڴ��ݶ�����ò������෵��ֵ
	void return_params(std::string&, std::string&, int&);

	//	��������෵��ֵ
	std::array<std::string, 2> return_array();

	//	�����Զ���Ķ෵��ֵ
	std::tuple<std::string, std::string, int> return_tuple();

	//	�෵��ֵ������1��struct��2���������ò����ٸ�ֵ��3���������飻4��tuple��������ͬ����ֵ��
	void init_return();

	//	template����ͨ��ָ��������������ν�ĺ������ض���
	template<typename first_param>
	void template1(first_param param);

	//	template��������ı������ͺ������С
	template<typename _arr, int size>
	class sarray {
	private:
		_arr arr[size];
	public:
		int getSize() const;
	};

	void initTemplate();

	template<typename _value>
	//	����β��ﶨ��Ļص��������������ͻᵼ��lambda�޷�ʹ��[]����������������ᱨ�����������
	//void each(const std::vector<Value>& values, void(*handler)(Value));
	//	�β����ñ�׼�ⷽ��ģ�嶨��ص��������ͣ�lambda����ʹ��[]�������������
	void each(const std::vector<_value>&, const std::function<void(_value)>&);

	void init_auto();

	void do_work();

	void init_thread();

	void init_sort();

	void init_get_file();

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
