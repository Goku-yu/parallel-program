#include <string>
#include <cstring>
#include <iostream>
#include <unordered_map>
#include <queue>
#include <omp.h>
#include <cuda_runtime.h> 
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
// class PriorityQueue
// {
// public:
//     // 用vector实现的priority queue
//     vector<PT> priority;

//     // 模型作为成员，辅助猜测生成
//     model m;

//     // 计算一个pt的概率
//     void CalProb(PT &pt);

//     // 优先队列的初始化
//     void init();

//     // 对优先队列的一个PT，生成所有guesses
//     void Generate(PT pt);

//     // 将优先队列最前面的一个PT
//     void PopNext();
//     int total_guesses = 0;
//     vector<string> guesses;
// };


// class PriorityQueue
// {
// public:
//     // 使用vector容器模拟优先队列
//     vector<PT> priority;

//     // 模型成员变量，辅助生成口令猜测
//     model m;

//     // 计算指定PT的概率
//     void CalProb(PT& pt);

//     // 初始化队列
//     void init();

//     // 针对一个PT生成对应的口令猜测
//     void Generate(PT pt);

//     // 移除优先队列顶部的PT并进行处理
//     void PopNext();

//     int total_guesses = 0;
//     vector<string> guesses;

//     // 一次性处理batch_size个PT（弹出、生成、拓展）
//     void PopBatchAndGenerate(int batch_size)
//     {
//         vector<PT> batch;

//         // 第一阶段：从队列尾部取出batch_size个PT
//         for (int count = 0; count < batch_size && !priority.empty(); ++count) {
//             PT current = priority.back();
//             priority.pop_back();
//             batch.emplace_back(current);
//         }

//         // 第二阶段：为批次中每个PT生成猜测字符串
//         for (PT& pt : batch) {
//             Generate(pt);
//         }

//         // 第三阶段：扩展生成新PT并放回优先队列
//         for (PT& pt : batch) {
//             vector<PT> expanded = pt.NewPTs();
//             for (PT& child : expanded) {
//                 priority.emplace_back(child);
//             }
//         }
//     }
// };

// //GPU进阶1
// extern "C" void gpu_generate(char *flat_input,int count,const std::string &prefix,std::vector<std::string> &result);

// class PriorityQueue
// {
// public:
//     // 使用vector容器模拟优先队列
//     vector<PT> priority;

//     // 模型成员变量，辅助生成口令猜测
//     model m;

//     // 计算指定PT的概率
//     void CalProb(PT& pt);

//     // 初始化队列
//     void init();

//     // 针对一个PT生成对应的口令猜测
//     void Generate(PT pt);

//     // 移除优先队列顶部的PT并进行处理
//     void PopNext();

//     int total_guesses = 0;
//     vector<string> guesses;

//     // 一次性处理batch_size个PT（弹出、生成、拓展）
       
//     void PopBatchAndGenerate(int batch_size)
//     {
//         vector<PT> batch;

//         // 第一阶段：从队列尾部取出batch_size个PT
//         for (int count = 0; count < batch_size && !priority.empty(); ++count) {
//             PT current = priority.back();
//             priority.pop_back();
//             batch.emplace_back(current);
//         }

//         // 第二阶段：批量为每个PT生成猜测字符串（合并到 GPU 调用中）
//         const int MAX_LEN = 64;
//         int total_input_count = 0;
//         std::vector<char> flat_input; // 所有PT的输入拼在一起
//         std::vector<int> start_pos;   // 每个PT的flat_input起始位置
//         std::vector<std::string> all_prefix;

//         for (PT& pt : batch) {
//             CalProb(pt);

//             segment *a = nullptr;
//             int last = pt.content.size() - 1;
//             const segment &seg = pt.content[last];
//             if (seg.type == 1) a = &m.letters[m.FindLetter(seg)];
//             else if (seg.type == 2) a = &m.digits[m.FindDigit(seg)];
//             else if (seg.type == 3) a = &m.symbols[m.FindSymbol(seg)];
//             else continue;

//             const std::vector<std::string>& values = a->ordered_values;
//             int count = values.size();
//             total_input_count += count;

//             std::string prefix = "";
//             if (pt.content.size() > 1) {
//                 int seg_idx = 0;
//                 for (int idx : pt.curr_indices) {
//                     if (pt.content[seg_idx].type == 1)
//                         prefix += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
//                     else if (pt.content[seg_idx].type == 2)
//                         prefix += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
//                     else if (pt.content[seg_idx].type == 3)
//                         prefix += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];

//                     seg_idx++;
//                     if (seg_idx == pt.content.size() - 1)
//                         break;
//                 }
//             }

//             // 保存当前PT的prefix与起始位置
//             start_pos.push_back(flat_input.size());
//             all_prefix.push_back(prefix);

//             for (const std::string& val : values) {
//                 char buffer[MAX_LEN] = {0};
//                 strncpy(buffer, val.c_str(), MAX_LEN - 1);
//                 flat_input.insert(flat_input.end(), buffer, buffer + MAX_LEN);
//             }
//         }

//         // 批量调用 GPU（注意合并 prefix 与 inputs 可能需要额外设计）
//         // 这里只演示调用一次（建议改写 gpu_generate 支持 vector<string> prefix）
//         for (size_t i = 0; i < batch.size(); ++i) {
//             int start = start_pos[i] / MAX_LEN;
//             int end = (i + 1 < batch.size() ? start_pos[i + 1] / MAX_LEN : total_input_count);
//             int count = end - start;

//             if (count <= 0) continue;

//             std::vector<std::string> results;
//             gpu_generate(&flat_input[start * MAX_LEN], count, all_prefix[i], results);

//             for (const auto& s : results) {
//                 guesses.emplace_back(s);
//                 total_guesses++;
//             }
//         }

//         // 第三阶段：扩展生成新PT并放回优先队列
//         for (PT& pt : batch) {
//             vector<PT> expanded = pt.NewPTs();
//             for (PT& child : expanded) {
//                 priority.emplace_back(child);
//             }
//         }
//     }
// };


// //创新实验一

// extern "C" void gpu_generate(char *flat_input,int count,const std::string &prefix,std::vector<std::string> &result);

// int get_mpi_rank() ;

// class PriorityQueue
// {
// public:
//     // 使用vector容器模拟优先队列
//     vector<PT> priority;

//     // 模型成员变量，辅助生成口令猜测
//     model m;

//     // 计算指定PT的概率
//     void CalProb(PT& pt);

//     // 初始化队列
//     void init();

//     // 针对一个PT生成对应的口令猜测
//     void Generate(PT pt);

//     // 移除优先队列顶部的PT并进行处理
//     void PopNext();

//     int total_guesses = 0;
//     vector<string> guesses;

//     // 一次性处理batch_size个PT（弹出、生成、拓展）
       
    

//     void PopBatchAndGenerate(int batch_size) {
//         vector<PT> batch;
//         for (int count = 0; count < batch_size && !priority.empty(); ++count) {
//             PT current = priority.back();
//             priority.pop_back();
//             batch.emplace_back(current);
//         }

//         const int MAX_LEN = 64;
//         int total_input_count = 0;
//         std::vector<char> flat_input;
//         std::vector<int> start_pos;
//         std::vector<std::string> all_prefix;

//         for (PT& pt : batch) {
//             CalProb(pt);
//             segment *a = nullptr;
//             int last = pt.content.size() - 1;
//             const segment &seg = pt.content[last];
//             if (seg.type == 1) a = &m.letters[m.FindLetter(seg)];
//             else if (seg.type == 2) a = &m.digits[m.FindDigit(seg)];
//             else if (seg.type == 3) a = &m.symbols[m.FindSymbol(seg)];
//             else continue;

//             const std::vector<std::string>& values = a->ordered_values;
//             int count = values.size();
//             total_input_count += count;

//             std::string prefix = "";
//             if (pt.content.size() > 1) {
//                 int seg_idx = 0;
//                 for (int idx : pt.curr_indices) {
//                     if (pt.content[seg_idx].type == 1)
//                         prefix += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
//                     else if (pt.content[seg_idx].type == 2)
//                         prefix += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
//                     else if (pt.content[seg_idx].type == 3)
//                         prefix += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
//                     if (++seg_idx == pt.content.size() - 1) break;
//                 }
//             }

//             start_pos.push_back(flat_input.size());
//             all_prefix.push_back(prefix);

//             for (const std::string& val : values) {
//                 char buffer[MAX_LEN] = {0};
//                 strncpy(buffer, val.c_str(), MAX_LEN - 1);
//                 flat_input.insert(flat_input.end(), buffer, buffer + MAX_LEN);
//             }
//         }

//         int rank = get_mpi_rank(); // MPI进程编号
//         cudaSetDevice(rank);       // 为该进程绑定唯一 GPU

//         for (size_t i = 0; i < batch.size(); ++i) {
//             int start = start_pos[i] / MAX_LEN;
//             int end = (i + 1 < batch.size() ? start_pos[i + 1] / MAX_LEN : total_input_count);
//             int count = end - start;
//             if (count <= 0) continue;

//             std::vector<std::string> results;
//             gpu_generate(&flat_input[start * MAX_LEN], count, all_prefix[i], results);
//             for (const auto& s : results) {
//                 guesses.emplace_back(s);
//                 total_guesses++;
//             }
//         }

//         for (PT& pt : batch) {
//             vector<PT> expanded = pt.NewPTs();
//             for (PT& child : expanded) {
//                 priority.emplace_back(child);
//             }
//         }
//     }
// };

//创新实验二
extern "C" void gpu_generate(char *flat_input,int count,const std::string &prefix,std::vector<std::string> &result);

int get_mpi_rank() ;

class PriorityQueue
{
public:
    // 使用vector容器模拟优先队列
    vector<PT> priority;

    // 模型成员变量，辅助生成口令猜测
    model m;

    // 计算指定PT的概率
    void CalProb(PT& pt);

    // 初始化队列
    void init();

    // 针对一个PT生成对应的口令猜测
    void Generate(PT pt);

    // 移除优先队列顶部的PT并进行处理
    void PopNext();

    int total_guesses = 0;
    vector<string> guesses;

    // 一次性处理batch_size个PT（弹出、生成、拓展）
       
    


    void PopBatchAndGenerate(int batch_size)
    {
        vector<PT> batch;

        // 第一阶段：从队列尾部取出 batch_size 个 PT
        for (int count = 0; count < batch_size && !priority.empty(); ++count) {
            PT current = priority.back();
            priority.pop_back();
            batch.emplace_back(current);
        }

        const int MAX_LEN = 64;

        // 第二阶段：并行为每个 PT 调用 gpu_generate
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < batch.size(); ++i) {
            PT &pt = batch[i];
            CalProb(pt);

            segment *a = nullptr;
            int last = pt.content.size() - 1;
            const segment &seg = pt.content[last];
            if (seg.type == 1) a = &m.letters[m.FindLetter(seg)];
            else if (seg.type == 2) a = &m.digits[m.FindDigit(seg)];
            else if (seg.type == 3) a = &m.symbols[m.FindSymbol(seg)];
            else continue;

            const std::vector<std::string>& values = a->ordered_values;
            int count = values.size();

            // 构造 prefix
            std::string prefix = "";
            if (pt.content.size() > 1) {
                int seg_idx = 0;
                for (int idx : pt.curr_indices) {
                    if (pt.content[seg_idx].type == 1)
                        prefix += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
                    else if (pt.content[seg_idx].type == 2)
                        prefix += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
                    else if (pt.content[seg_idx].type == 3)
                        prefix += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
                    if (++seg_idx == pt.content.size() - 1) break;
                }
            }

            // 构造 flat_input
            std::vector<char> flat_input(count * MAX_LEN, 0);
            for (int j = 0; j < count; ++j) {
                strncpy(&flat_input[j * MAX_LEN], values[j].c_str(), MAX_LEN - 1);
            }

            // GPU 并行拼接
            std::vector<std::string> results;
            gpu_generate(flat_input.data(), count, prefix, results);

            // 合并结果到共享 guesses 向量
            #pragma omp critical
            {
                for (const auto& s : results) {
                    guesses.emplace_back(s);
                    total_guesses++;
                }
            }

            // 将当前 PT 扩展并放回队列
            std::vector<PT> expanded = pt.NewPTs();
            #pragma omp critical
            for (PT& child : expanded) {
                priority.emplace_back(child);
            }
        }
    }
};
