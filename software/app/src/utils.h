#pragma once
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
		explicit Player(PlayerLevel level);
		~Player();
		void move(int new_x, int new_y);
	};
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
		Vec vec;
		Vecv();
		Vec& getVec();
	};
	void initCalculate();
}