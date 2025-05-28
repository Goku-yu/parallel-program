#include <string>
#include <iostream>
#include <unordered_map>
#include <queue>

#include <semaphore.h>  // 解决 sem_t 未定义的问题
#include <pthread.h>    // 确保 pthread 相关函数可用
#include <vector>
#include <string>

// #include <chrono>   
// using namespace chrono;
using namespace std;

class segment
{
public:
    int type; // 0: 未设置, 1: 字母, 2: 数字, 3: 特殊字符
    int length; // 长度，例如S6的长度就是6
    segment(int type, int length)
    {
        this->type = type;
        this->length = length;
    };

    // 打印相关信息
    void PrintSeg();

    // 按照概率降序排列的value。例如，123是D3的一个具体value，其概率在D3的所有value中排名第三，那么其位置就是ordered_values[2]
    vector<string> ordered_values;

    // 按照概率降序排列的频数（概率）
    vector<int> ordered_freqs;

    // total_freq作为分母，用于计算每个value的概率
    int total_freq = 0;

    // 未排序的value，其中int就是对应的id
    unordered_map<string, int> values;

    // 根据id，在freqs中查找/修改一个value的频数
    unordered_map<int, int> freqs;


    void insert(string value);
    void order();
    void PrintValues();
};

class PT
{
public:
    // 例如，L6D1的content大小为2，content[0]为L6，content[1]为D1
    vector<segment> content;

    // pivot值，参见PCFG的原理
    int pivot = 0;
    void insert(segment seg);
    void PrintPT();

    // 导出新的PT
    vector<PT> NewPTs();

    // 记录当前每个segment（除了最后一个）对应的value，在模型中的下标
    vector<int> curr_indices;

    // 记录当前每个segment（除了最后一个）对应的value，在模型中的最大下标（即最大可以是max_indices[x]-1）
    vector<int> max_indices;
    // void init();
    float preterm_prob;
    float prob;
};

class model
{
public:
    // 对于PT/LDS而言，序号是递增的
    // 训练时每遇到一个新的PT/LDS，就获取一个新的序号，并且当前序号递增1
    int preterm_id = -1;
    int letters_id = -1;
    int digits_id = -1;
    int symbols_id = -1;
    int GetNextPretermID()
    {
        preterm_id++;
        return preterm_id;
    };
    int GetNextLettersID()
    {
        letters_id++;
        return letters_id;
    };
    int GetNextDigitsID()
    {
        digits_id++;
        return digits_id;
    };
    int GetNextSymbolsID()
    {
        symbols_id++;
        return symbols_id;
    };

    // C++上机和数据结构实验中，一般不允许使用stl
    // 这就导致大家对stl不甚熟悉。现在是时候体会stl的便捷之处了
    // unordered_map: 无序映射
    int total_preterm = 0;
    vector<PT> preterminals;
    int FindPT(PT pt);

    vector<segment> letters;
    vector<segment> digits;
    vector<segment> symbols;
    int FindLetter(segment seg);
    int FindDigit(segment seg);
    int FindSymbol(segment seg);

    unordered_map<int, int> preterm_freq;
    unordered_map<int, int> letters_freq;
    unordered_map<int, int> digits_freq;
    unordered_map<int, int> symbols_freq;

    vector<PT> ordered_pts;

    // 给定一个训练集，对模型进行训练
    void train(string train_path);

    // 对已经训练的模型进行保存
    void store(string store_path);

    // 从现有的模型文件中加载模型
    void load(string load_path);

    // 对一个给定的口令进行切分
    void parse(string pw);

    void order();

    // 打印模型
    void print();
};

// 优先队列，用于按照概率降序生成口令猜测
// 实际上，这个class负责队列维护、口令生成、结果存储的全部过程
class PriorityQueue
{
public:
    // 用vector实现的priority queue
    vector<PT> priority;

    // 模型作为成员，辅助猜测生成
    model m;

    // 计算一个pt的概率
    void CalProb(PT &pt);

    // 优先队列的初始化
    void init();

    // 对优先队列的一个PT，生成所有guesses
    void Generate(PT pt);

    // 将优先队列最前面的一个PT
    void PopNext();
    int total_guesses = 0;
    vector<string> guesses;
    
    ~PriorityQueue() {
        // 通知线程退出
        for (int i = 0; i < THREAD_NUM; ++i) {
            params1[i].start_idx = -1;  // 特殊值表示退出
            sem_post(&start_sem1[i]);
            pthread_join(threads1[i], nullptr);
            
            params2[i].start_idx = -1;  // 特殊值表示退出
            sem_post(&start_sem2[i]);
            pthread_join(threads2[i], nullptr);
            
            sem_destroy(&start_sem1[i]);
            sem_destroy(&done_sem1[i]);
            sem_destroy(&start_sem2[i]);
            sem_destroy(&done_sem2[i]);
        }
    }

private:
    // 添加这些成员变量
    static const int THREAD_NUM = 4; // 或者你想要的线程数
    bool threads_initialized = false;
    
    // 线程相关变量
    pthread_t threads1[THREAD_NUM];
    pthread_t threads2[THREAD_NUM];
    sem_t start_sem1[THREAD_NUM];
    sem_t done_sem1[THREAD_NUM];
    sem_t start_sem2[THREAD_NUM];
    sem_t done_sem2[THREAD_NUM];
    
    // 线程参数结构体
    // 在 PriorityQueue 类中添加以下结构体定义
    struct ThreadParams1 {
        segment* segment_ptr;
        int start_idx;
        int end_idx;
        int thread_id;
        std::vector<std::string>* local_results;
        sem_t* start_sem;
        sem_t* done_sem;
        
        // 添加构造函数
        ThreadParams1() : segment_ptr(nullptr), start_idx(0), end_idx(0), thread_id(0), 
                        local_results(nullptr), start_sem(nullptr), done_sem(nullptr) {}
    };

    struct ThreadParams2 {
        std::string base_string;
        segment* segment_ptr;
        int start_idx;
        int end_idx;
        int thread_id;
        std::vector<std::string>* local_results;
        sem_t* start_sem;
        sem_t* done_sem;
        
        // 添加构造函数
        ThreadParams2() : base_string(""), segment_ptr(nullptr), start_idx(0), end_idx(0),
                        thread_id(0), local_results(nullptr), start_sem(nullptr), done_sem(nullptr) {}
    };
    
    ThreadParams1 params1[THREAD_NUM];
    ThreadParams2 params2[THREAD_NUM];
    std::vector<std::string> local_results1[THREAD_NUM];
    std::vector<std::string> local_results2[THREAD_NUM];
    
    // 声明线程函数为静态成员函数
    static void* thread_func1(void* arg);
    static void* thread_func2(void* arg);

};
