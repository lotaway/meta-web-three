/*
* �˴����ԭ�������ģʽ��ɱ�Ҫ�ԣ���Ҫ���ó�����Ա����ȶ���仯��ƽ�⣬����ʱ���ȶ��ģ��仯��ͨ������ʱ���ݲ�ͬ���仯��
* 1. �߲�ʵ�ֲ��������ڵײ�ʵ�֣���ֻ�������ײ����
* 2. ������������ʵ�֣�����ʵ����������
*/

// ģ�巽�� Template Method��ͨ�������˹��̣��ṩ���뷽���������󶨵ĵ��÷�ʽ����������Զ�����Ӧ�ó��򿪷������̳�ʵ�ֵķ�������
#include <stack>
#include "logger.h"
#include "utils.h"

class Library {
protected:
	Library() {
		init();
	}
	~Library() {
		on_destroy();
	}
	virtual void run() {
		on_create();
	}
	virtual void init() {

	}
	virtual void on_create() = 0;
	virtual void on_destroy() {

	}
};

//	����ģʽ Strategy Mode����Ҫ���ڼ���if else/switch�Դ�����ͬ����Ĵ���ͨ�����󡢷����ⲿ�ֹ����������̱�ø���ģ�廯
struct IStrategy {
public:
	virtual bool deal(const char& order_number) = 0;
};

class ChinaStrategy : public IStrategy {
public:
	bool deal(const char& order_number) override {
		return true;
	}
};

class EnglandStrategy : public IStrategy {
public:
	bool deal(const char& order_number) override {
		return false;
	}
};

class OrderController {
private:
	IStrategy* strategy;
	const char *order_number;
public:
	OrderController(IStrategy*_strategy, const char* _order_number) : strategy(_strategy), order_number(_order_number) {

	}
	~OrderController() {
		delete strategy;
	}
	void deal() {
		bool result = strategy->deal(*order_number);
	}
};

void application() {
	IStrategy*cs = new ChinaStrategy();
	OrderController oc1(cs, "123");
	IStrategy* es = new EnglandStrategy();
	OrderController oc2(es, "456");
}

//	�۲���ģʽ Obverser Mode�����ڴ���������ֱ�ӹ�������֮����ض����Ƶ�ʵ�֣�����֪ͨ�����ȸ��µȣ�����������Ҫ�����������У�Ҳ��������Ҫ������ĳ������Ľ��棬Ҳ��������֪ͨ�����ⲻ�����������йص��඼ȥֱ�Ӽ̳У�����ʹ�ýӿ�����Ҫʱȥʵ��

class Activity {
	virtual void on_create() = 0;
	virtual void on_destroy() {

	}
};

class IProgress {
public:
	virtual void update_progress(float present) = 0;
};

class NotificationBar {
public:
	void doUpdate() {

	}
};

//	�����棬�����ļ�����֮��Ҫͨ��ʵ�ֽӿ����������ؽ��Ȳ�����֪ͨ��
class MainActivity: public Activity, public IProgress {
	NotificationBar* nBar;
	MainActivity(NotificationBar* _nBar): nBar(_nBar) {

	}
	~MainActivity() {
		delete nBar;
	}
	void on_create() override {

	}
	// ʵ�ֹ۲촦��
	void update_progress(float present) override {
		//	���ݷ����ٷֱȸ���֪ͨ��
		nBar->doUpdate();
	}
};

//	�����б���棬�����ļ����ؽ��Ȳ����½���
class DownloadListActivity : public Activity, public IProgress {
	void on_create() override {

	}
	//	ʵ�ֹ۲촦��
	void update_progress(float present) override {
		// ���ݷ����ٷֱȸ��½���
		//this->update_view();
	}
};

//	��������
class Task {
private:
	const char* name;
protected:
	Task(const char* _name) : name(_name) {

	}
	virtual void start() = 0;
	//virtual void pause() = 0;
	//virtual void resume() = 0;
	virtual void stop() = 0;
	virtual void onProgress() {
	}
};

//	�ļ���������
class DownloadFileTask: public Task {
private:
	const char* file_path;
	IProgress* iProgress;
	float present;
public:
	DownloadFileTask(const char* _file_path, IProgress* _iProgress) : Task(_file_path), file_path(_file_path), iProgress(_iProgress) {
		present = 0;
	}
	~DownloadFileTask() {
		delete file_path;
		delete iProgress;
	}
	void start() override {

	}
	void stop() override {

	}
	//	����������ȸ���ʱ��Ҳ֪ͨ���ӵĹ۲���
	void onProgress() {
		iProgress->update_progress(present);
	}
};


// װ��ģʽ Decorator Mode����Ҫ��ͨ���̳ж�������ԭ�з�������������߰�����������ͬʱִ�С�

class IVehicle {
public:
	virtual void move() = 0;
	virtual void stop() = 0;
};

class Car : public IVehicle {
public:
	void move() override {
		utils::print_string("car run");
	}
	void stop() override {
		utils::print_string("car stop");
	}
};

class Airplane : public IVehicle {
public:
	void move() override {
		utils::print_string("airplane run");
	}
	void stop() override {
		utils::print_string("airplane stop");
	}
};

class IRobot {
public:
	virtual void move() = 0;
	virtual void speak(std::string sentence) {
		utils::print_string(sentence);
	}
};

//	��һ��װ��������չ�˻����˵�����
class ITransformer : public IRobot {
protected:
	IVehicle* vehicle;
	const char* name;
	int state = 0;
public:
	ITransformer(const char* nam) : name(nam) {

	}
	ITransformer(const char* nam, IVehicle* ve) : name(nam), vehicle(ve) {

	}
	~ITransformer() {
		delete vehicle;
		delete name;
	}
	//	���°�װ����Ϊ
	virtual void move() override {
		vehicle->move();
		this->speak("transformer moving");
	}
	//	��������������ӵ�и��๦��
	virtual void transform() {
		if (state == 0) {
			utils::print_string("transform to vehicle");
			state = 1;
		}
		else {
			utils::print_string("transform to robot");
			state = 0;
		}
	}
};

//	�ڶ���װ��������չ�˱��ν�յ�����
class ITrichanger : public ITransformer {
	IVehicle* sec_vehicle;
	~ITrichanger() {
		ITransformer::~ITransformer();
		delete sec_vehicle;
	}
	//	��������������ӵ�и��๦��
	void transform(int st) {
		switch (st) {
		case 0:
			transform_to_first_vehicle();
			break;
		case 1:
			transform_to_second_vehicle();
			break;
		case 2:
			transform_to_robot();
			break;
		default:
			break;
		}
	}
	void transform_to_first_vehicle() {
		utils::print_string("transform_to_first_vehicle");
		state = 0;
	}
	void transform_to_second_vehicle() {
		utils::print_string("transform_to_second_vehicle");
		state = 1;
	}
	void transform_to_robot() {
		utils::print_string("transform_to_robot");
		state = 2;
	}
};

class Autobot : ITransformer {
public:
	Autobot(const char* name): ITransformer(name) {}
};

class Decepticon : ITransformer {

};



// ��ģʽ Bridge Mode����װ����ģʽ�ǳ����ƣ��������Ž���ζ�������ڲ�����������������蹦��

class IChange {
public:
	virtual void change() = 0;
};

class Transformer {
	IChange* iChange;
public:
	Transformer(IChange* ic): iChange(ic) {

	}
	~Transformer() {
		delete iChange;
	}
	//	ע��ʵ�ַ�ʽ�ǽ�����������ɣ���װ����һ���Ƕ�������������ǿ��������
	void transform() {
		return iChange->change();
	}
};

// ����ģʽ Factory Mode�������������ͣ��������������󹤳���ԭ�͡�������

class IFactoryMethod {
public:
	virtual void config(const std::string, const std::string) = 0;
	virtual IVehicle* create() = 0;
};

//	��������, Factory Method��Ŀ���ǰ�������ָ����
class FactoryMethod: public IFactoryMethod {
public:
	void config(const std::string key, const std::string value) override {

	}
	IVehicle* create() override {
		return new Car();
	}
};

void testFactoryMethod() {
	FactoryMethod fm;
	IVehicle* vehicle = fm.create();
}

//	���󹤳�, Abstract Factory��Ŀ���������й�����һ������İ�װ��
class IHouse {
protected:
	std::string* address;
	IVehicle* vehicle;
	ITransformer* transformer;
public:
	IHouse(std::string* addr, IVehicle* veh, ITransformer* tf): address(addr), vehicle(veh), transformer(tf) {}
	IHouse(const IHouse& other) {

	}
	~IHouse() {
		delete address;
		delete vehicle;
		delete transformer;
	}
	virtual void come_back() = 0;
	virtual void go_out() = 0;
	virtual void interactive() = 0;
};

//	ʵ��ʹ��ʱ�Գ��󹤳�Ҫ��ĸ��ֳ��������໯
class MyHouse : public IHouse {
public:
	MyHouse(std::string* address, Car* vehicle, ITrichanger* transform): IHouse(address, vehicle, transform) {

	}
	void come_back() override {
		this->vehicle->move();
	}
	void go_out() override {
		this->vehicle->move();
	}
	void interactive() override {
		this->transformer->speak("hello");
	}
};

class Me {
	IHouse* house;
	void go_home() {
		house->come_back();
	}
	void go_work() {
		house->go_out();
	}
	void interactive() {
		utils::print_string("what?");
		house->interactive();
	}
};

//	ԭ�ͣ�Prototype Mode����Ҫ�ǽ�����������ʱ��ͨ���ȴ���һ�Σ�����������Ĳ��������ԣ�֮������ظ����и��ƴ�����ʵ��
class IFactoryPrototype {
	IFactoryMethod* factoryMethod;
public:
	IFactoryPrototype(IFactoryMethod* fm) : factoryMethod(fm) {};
	~IFactoryPrototype() {
		delete factoryMethod;
	}
	virtual IFactoryMethod* clone() = 0;
};

//	��������builder����Ҫ���ڽ�ĳ����Ĵ������̽��з�װ��������������ʹ�ã�������Ҫ�޸�ʱ���趯��ԭ�����ࣨ�ϲ��ķ�ʽ����ֱ����ԭ�����������init������
class IFactoryBuilder {
	IFactoryMethod* factoryMethod;
public:
	IFactoryBuilder(IFactoryMethod* fm) : factoryMethod(fm) {

	}
	~IFactoryBuilder() {
		delete factoryMethod;
	}
	IFactoryMethod* build() {
		factoryMethod->config("do", "something");
		factoryMethod->create();
		return factoryMethod;
	}
};

// ����ģʽ��Singleton������ȫ�ֻ���ϵͳ������Ϊ������Ӧ����ֻ��ҪΨһһ���ö��󣬷�����������ͬ�����ֲ����㱣��ö���ʵ��������Ҫ�ӳٱ��棬��˲��õ���ģʽ��ȷ����ͬ�ط��ĵ��ö����õ�Ψһһ��ʵ����

class Singleton {
	//	˽�л����캯����ֹ����ʵ��
	Singleton();
	//	˽�л����ƹ��캯����ֹ����
	Singleton(const Singleton& other);
public:
	static Singleton* instance;
	//	ָ����û��ʵ��ʱ�Ž��д���������ֱ�ӷ���ԭ��ʵ������
	static Singleton* getInstance() {
		if (instance == nullptr)
			instance = new Singleton;
		return instance;
	}
};

//	��Ԫģʽ��Flyweight Mode����Ҫ����ɹ��������Դ��ά��һ�����޵ĳ���������Ҫ��������ʱ���ڴ˴����������Ż��ж�ԭ�в�ʹ�õĶ������ɾ�������ڴ��ƶ�����ȷ����Դ�����˷ѡ�
class AutobotFactory {
private:
	std::unordered_map<const char*, Autobot*> autobotPool;
public:
	Autobot* getAutobot(const char* name) {
		std::unordered_map<const char*, Autobot*>::iterator it = autobotPool.find(name);
		if (it == autobotPool.end()) {
			autobotPool[name] = new Autobot(name);
		}
		return autobotPool[name];
	}
};

//	����ģʽ������˵�����ģʽ������˵�����ԭ����Ҫ�����ڽ���ӿ�֮���������⣬����һ��ʼ����ֱ�Ӱ�ť��Ӧ��·��Ӳ�����ܣ�֮�����������Ӳ������֮��ʼ��������Э�顢�̼�������ϵͳ���������ƽ̨��ȣ���Щ����ͨ������;ۺ�Լ���ӿ�֮���ͨѶ��ʽ��ʹ֮���
// ����������ģʽ�� Facade��Proxy��Adapter��Mediator����Ʒ�ʽֻҪ�������Ҫ�󼴿ɡ�

// Facade���Ӽܹ���ο���ϵͳ���ڲ�������һϵ�й�����ǿ������������ṩ�����޵���Ӧ�ӿڷ����ȸ��ⲿʹ�á�

class IConnect {
public:
	virtual bool connect() = 0;
	virtual bool disconnect() = 0;
};

class ICommand {
protected:
	IConnect* connect;
public:
	virtual void exec(const std::string) = 0;
	virtual std::string version() = 0;
};

class ISql {
protected:
	IConnect* connect;
public:
	virtual void query(const std::string) = 0;
};

class DBClient {
	ICommand* cmd;
	IConnect* connect;
	ISql* sql;
public:
	virtual void init() {
		connect->connect();
		utils::print_string(cmd->version());
	}
};

// Proxy�������ܱ��ֶ�ԭ�ж��󲻱䣬���ֽ�һ�������Ż����Ȩ�޵ĸ����������ȷ��ʹ�ò������ţ�����ֲ�ʽ��Ҫ����rpcԶ�̵��ã���ͨ�������������ַ�ʽ�����ͨ�Ĵ�������ã������һ�ִ���

class IRpc {
public:
	virtual bool auth(const std::string) = 0;
	virtual std::string invoke() = 0;
};

template <typename ResponseData>
class IService {
public:
	virtual ResponseData request() = 0;
};

class ServiceProxy : IService<bool> {
	IRpc* rpc;
public:
	virtual bool request() {
		if (rpc->auth("access_key")) {
			std::string result = rpc->invoke();
			if (result != "")
				return true;
		}
		return false;
	}
};

class CommpilerApplication : public Library {
public:
	void run() {
		ServiceProxy* serviceProxy = new ServiceProxy;
		bool result = serviceProxy->request();
	}
protected:
	/*void run() override {

	}*/
	void on_create() {

	}
};

// Adapter��������Ҳ�Ƿǳ���������Ʒ�ʽ����Ҫ���ö�����Ӧ�����֮���ܸ��õء��򵥵ضԽ���ȥ��������Ҫ����Щ�౾����и��졣

class IFile {
public:
	std::string name;
	std::string path;
};

class ISystem {
public:
	virtual IFile* readFile(const std::string) = 0;
	virtual bool writeFile(const std::string) = 0;
	virtual IFile* createFile(const std::string) = 0;
};

class ICore {
public:
	virtual IFile* file(const std::string) = 0;
};

class Adapter : ICore {
	ISystem* system;
public:
	IFile* file(const std::string path) override {
		IFile* file = system->readFile(path);
		if (file == nullptr) {
			return system->createFile(path);
		}
		return file;
	}
};

//	Mediator���н��ߣ���Ҫ�Ƿ����������ǿ��Ϲ�ϵ������ǰ�����Ӵ���ʹ�õ�ISystem��ICoreҲ��һ���н��ߣ�������������System��Core����Ϲ�ϵ����Ȼ�󲿷��н��߻��и��Ӷ����Ĺ������ԣ�������ɸ���Э�������幦�ܡ�

struct IMessage {
	std::string key;
	std::string data;
};

//	��Ϣ���о���һ���н��ߣ���һ��������Ӹ������塢�۶ϡ������ȴ�ʩ
class IMessageQueue {
	std::stack<IMessage> messages;
public:
	virtual void add(const std::string key, const std::string data) = 0;
	virtual void del(const std::string key) = 0;
};

class IServiceA {
	IMessageQueue* mq;
public:
	virtual void run() {
		mq->add("ServiceA", "active");
	}
};

class IServiceB {
	IMessageQueue* mq;
public:
	virtual void run() {
		mq->add("ServiceB", "active");
	}
};

//	״̬ģʽ��State Mode�� ͨ����״̬�ı����˲�ͬ����ʵ���ı仯��������״̬�޸�ʱ�ľ������

class IStatus {
protected:
	IStatus();
	IStatus(IStatus& other);
	static IStatus* m_instance;
public:
	IStatus* nextStatus;
	virtual void settle() = 0;
	virtual void cancel() = 0;
	virtual void done() = 0;
};

class CancelStatus : public IStatus {
	static CancelStatus* m_instance;
public:
	static CancelStatus* getInstance() {
		if (m_instance == nullptr)
			m_instance = new CancelStatus;
		return m_instance;
	}
	void settle() override;
	void cancel() override;
	void done() override;
};

class DoneStatus : public IStatus {
	static DoneStatus* m_instance;
public:
	static DoneStatus* getInstance() {
		if (m_instance == nullptr)
			m_instance = new DoneStatus;
		return m_instance;
	}
	void settle() override;
	void cancel() override;
	void done() override;
};

class SettleStatus : public IStatus {
	static SettleStatus* m_instance;
public:
	static SettleStatus* getInstance() {
		if (m_instance == nullptr)
			m_instance = new SettleStatus;
		return m_instance;
	}
	void settle() override;
	void cancel() override {
		nextStatus = CancelStatus::getInstance();
	}
	void done() override {
		nextStatus = DoneStatus::getInstance();
	}
};

//	���ж���״̬����͸ı�״̬������ж϶��ǽ���״̬�������������ֻ������ã������κ����жϸ�ֵ�Ķ��⴦��
class IOrderInfo {
	IStatus* status;
public:
	IOrderInfo() {
		status = SettleStatus::getInstance();
	}
	~IOrderInfo() {
		delete status;
	}
	void cancel() {
		status->cancel();
		status = status->nextStatus;
	}
	void confirmReceviced() {
		status->done();
		status = status->nextStatus;
	}
};

// ����¼��Memento Mode����Ҫ�����ڽ�ʵ�������״̬����һ�α��ݱ��棬֮�󾭹�һϵ�еı仯����Խ��лָ����ִ�����һ���Դ��ж������л�����������ʹ��Ҫ�Զ���Ҳ�и�����ķ�ʽ�ɹ����壬����Ҫ���������ù��ܡ�
template <class T>
class IMemento {
public:
	virtual T* load() {
	}
	virtual void save(T * obj) {
	}
};

class TargetClass {
protected:
	int count = 0;
public:
	void requestFresh() {
		IMemento<TargetClass> memento;
		memento.save(this);
		// ...do something change this class status
		count++;
		//	reverse change
		TargetClass* origin = memento.load();
		count = origin->count;
	}
};

// ���ģʽ��Composite Mode��ͨ�����������ظ��Ľṹ����ɶ���������ϵݹ�ʹ�ã������ڱ������רע
class LeafComponent {
protected:
	std::string value;
public:
	LeafComponent(const std::string& val) : value(val) {}
};
class TreeNodeComponent {
public:
	LeafComponent* leaf;
	TreeNodeComponent* left;
	TreeNodeComponent* right;
	TreeNodeComponent(const std::string& value) : leaf(new LeafComponent{ value }) {}
	TreeNodeComponent(LeafComponent* l): leaf(l) {}
	TreeNodeComponent(LeafComponent* l, TreeNodeComponent* lef): leaf(l), left(lef) {}
	TreeNodeComponent(LeafComponent* l, TreeNodeComponent* lef, TreeNodeComponent* rig) : leaf(l), left(lef), right(rig) {}
	~TreeNodeComponent() {
		delete leaf;
		delete left;
		delete right;
	}
};

void testTreeNode() {
	TreeNodeComponent root{ "1"};
	root.left = new TreeNodeComponent{ "2" };
	root.right = new TreeNodeComponent{ "2" };
}

// ������ģʽ��Iterator Mode���ǳ������������б�ʽ���������ظ����ڲ����ݵ��Ż��㷨��ȡ���������Ҫ����ʱ��ֱ��ʹ�õ���������ȡֵ�������ȣ�������Ҫ����Ӧ����ζ�ȡ�Ͷ�ȡ�ĸ�ֵ��

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
	//logger::out(vector.size());
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

// ְ������Chain Mode��ͨ��������ʱ��̬�����¼�����������������������Ƿ����غʹ����¼������������¼��ߺʹ����¼��߿��Խ������Ҫ��ǿ���������߶��һ���������⡣
class IEventRequest {
public:
	const std::string name;
	IEventRequest(const std::string n): name(n) {}
};

class EventResponse {
protected:
	IEventRequest* event;
	EventResponse* next;
public:
	EventResponse(IEventRequest* request): event(request) {}
	EventResponse* setHandler(EventResponse* nex) {
		next = nex;
		return next;
	}
	virtual IEventRequest* handler(IEventRequest* request) {
		if (next != nullptr)
			return next->handler(request);
		return request;
	}
};

// ����ģʽ��Command Mode����һ����һϵ����Ҫִ�е���Ϊ���������з�װ���Ա���ʱ�ظ��Զ�����е���ִ����Ϊ
class ICommandMode {
public:
	std::vector<std::string> cmd;
	void addCmd(const std::string& c) {
		cmd.push_back(c);
	}
	void execute() {
		std::vector<std::string>::iterator it = cmd.begin();
		while (it != cmd.end()) {
			utils::print_string(*it);
			it++;
		}
	}
};

void testCommandMode() {
	ICommandMode cm;
	cm.addCmd("hello");
	cm.addCmd("is me.");
	cm.execute();
}

// ������ģʽ��Visitor Mode���ڲ���Ҫ�ظ��޸Ļ��������£��û��ౣ��һ�����������÷������Ա��Ժ����ֱ��ͨ�����ݷ������ķ�ʽ�øû����Լ����඼��ͨ��������ӵķ����������ػ����������

template <class T>
class IVisitor {
public:
	virtual void accept(T&) = 0;
};

class Target {
protected:
	IVisitor<Target>* visitor;
public:
	std::string name;
	void addVisitor(IVisitor<Target>* vis) {
		visitor = vis;
	}
	void show() {
		visitor->accept(*this);
	}
};

class VisitorA : public IVisitor<Target> {
public:
	void accept(Target& target) override {
		utils::print_string(target.name);
	}
};

class VisitorB : public IVisitor<Target> {
public:
	void accept(Target& target) override {
		std::cout << target.name << std::endl;
	}
};

// ������ģʽ���ǳ������ڱ��������༭����Ӳ��Э�����Ŀ����Ϊ��Ҫ�Զ�����е��﷨����ͬ��Ҳ��Ҫ�����Լ��Ľ������������������﷨�����Ա���ɷ�װ����⡢���䡢�Ż��ȹ���ʹ�á�

class IExpression {
public:
	const char* symbol;
	float value;
	IExpression* left;
	IExpression* right;
};

class AddExpression : public IExpression {
public:
	AddExpression(const std::string) {

	}
};

class MinusExpression : public IExpression {
public:
	MinusExpression(const std::string) {

	}
};

void analyse(std::string expStr) {
	if (expStr.find("+")) {
		AddExpression ast(expStr);
	}
	else if (expStr.find("-")) {
		MinusExpression ast(expStr);
	}
}