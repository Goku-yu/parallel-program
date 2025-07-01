#include "PCFG.h"
#include <pthread.h>
#include <unistd.h>
#include <mutex>
#include <vector>
#include <string>
#include <omp.h>
#include <sstream>
#include <cstring>
#include <thread>
#include <functional>      
#include <algorithm>
#include <atomic>
#include <numeric>
#include <future>
using namespace std;

void PriorityQueue::CalProb(PT &pt)
{
    // 计算PriorityQueue里面一个PT的流程如下：
    // 1. 首先需要计算一个PT本身的概率。例如，L6S1的概率为0.15
    // 2. 需要注意的是，Queue里面的PT不是“纯粹的”PT，而是除了最后一个segment以外，全部被value实例化的PT
    // 3. 所以，对于L6S1而言，其在Queue里面的实际PT可能是123456S1，其中“123456”为L6的一个具体value。
    // 4. 这个时候就需要计算123456在L6中出现的概率了。假设123456在所有L6 segment中的概率为0.1，那么123456S1的概率就是0.1*0.15

    // 计算一个PT本身的概率。后续所有具体segment value的概率，直接累乘在这个初始概率值上
    pt.prob = pt.preterm_prob;

    // index: 标注当前segment在PT中的位置
    int index = 0;


    for (int idx : pt.curr_indices)
    {
        // pt.content[index].PrintSeg();
        if (pt.content[index].type == 1)
        {
            // 下面这行代码的意义：
            // pt.content[index]：目前需要计算概率的segment
            // m.FindLetter(seg): 找到一个letter segment在模型中的对应下标
            // m.letters[m.FindLetter(seg)]：一个letter segment在模型中对应的所有统计数据
            // m.letters[m.FindLetter(seg)].ordered_values：一个letter segment在模型中，所有value的总数目
            pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq;
            // cout << m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.letters[m.FindLetter(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 2)
        {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
            // cout << m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.digits[m.FindDigit(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 3)
        {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].total_freq << endl;
        }
        index += 1;
    }
    // cout << pt.prob << endl;
}

void PriorityQueue::init()
{
    // cout << m.ordered_pts.size() << endl;
    // 用所有可能的PT，按概率降序填满整个优先队列
    for (PT pt : m.ordered_pts)
    {
        for (segment seg : pt.content)
        {
            if (seg.type == 1)
            {
                // 下面这行代码的意义：
                // max_indices用来表示PT中各个segment的可能数目。例如，L6S1中，假设模型统计到了100个L6，那么L6对应的最大下标就是99
                // （但由于后面采用了"<"的比较关系，所以其实max_indices[0]=100）
                // m.FindLetter(seg): 找到一个letter segment在模型中的对应下标
                // m.letters[m.FindLetter(seg)]：一个letter segment在模型中对应的所有统计数据
                // m.letters[m.FindLetter(seg)].ordered_values：一个letter segment在模型中，所有value的总数目
                pt.max_indices.emplace_back(m.letters[m.FindLetter(seg)].ordered_values.size());
            }
            if (seg.type == 2)
            {
                pt.max_indices.emplace_back(m.digits[m.FindDigit(seg)].ordered_values.size());
            }
            if (seg.type == 3)
            {
                pt.max_indices.emplace_back(m.symbols[m.FindSymbol(seg)].ordered_values.size());
            }
        }
        pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;
        // pt.PrintPT();
        // cout << " " << m.preterm_freq[m.FindPT(pt)] << " " << m.total_preterm << " " << pt.preterm_prob << endl;

        // 计算当前pt的概率
        CalProb(pt);
        // 将PT放入优先队列
        priority.emplace_back(pt);
    }
    // cout << "priority size:" << priority.size() << endl;
}

void PriorityQueue::PopNext()
{

    // 对优先队列最前面的PT，首先利用这个PT生成一系列猜测
    Generate(priority.front());

    // 然后需要根据即将出队的PT，生成一系列新的PT
    vector<PT> new_pts = priority.front().NewPTs();
    for (PT pt : new_pts)
    {
        // 计算概率
        CalProb(pt);
        // 接下来的这个循环，作用是根据概率，将新的PT插入到优先队列中
        for (auto iter = priority.begin(); iter != priority.end(); iter++)
        {
            // 对于非队首和队尾的特殊情况
            if (iter != priority.end() - 1 && iter != priority.begin())
            {
                // 判定概率
                if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob)
                {
                    priority.emplace(iter + 1, pt);
                    break;
                }
            }
            if (iter == priority.end() - 1)
            {
                priority.emplace_back(pt);
                break;
            }
            if (iter == priority.begin() && iter->prob < pt.prob)
            {
                priority.emplace(iter, pt);
                break;
            }
        }
    }

    // 现在队首的PT善后工作已经结束，将其出队（删除）
    priority.erase(priority.begin());
}

// 这个函数你就算看不懂，对并行算法的实现影响也不大
// 当然如果你想做一个基于多优先队列的并行算法，可能得稍微看一看了
vector<PT> PT::NewPTs()
{
    // 存储生成的新PT
    vector<PT> res;

    // 假如这个PT只有一个segment
    // 那么这个segment的所有value在出队前就已经被遍历完毕，并作为猜测输出
    // 因此，所有这个PT可能对应的口令猜测已经遍历完成，无需生成新的PT
    if (content.size() == 1)
    {
        return res;
    }
    else
    {
        // 最初的pivot值。我们将更改位置下标大于等于这个pivot值的segment的值（最后一个segment除外），并且一次只更改一个segment
        // 上面这句话里是不是有没看懂的地方？接着往下看你应该会更明白
        int init_pivot = pivot;

        // 开始遍历所有位置值大于等于init_pivot值的segment
        // 注意i < curr_indices.size() - 1，也就是除去了最后一个segment（这个segment的赋值预留给并行环节）
        for (int i = pivot; i < curr_indices.size() - 1; i += 1)
        {
            // curr_indices: 标记各segment目前的value在模型里对应的下标
            curr_indices[i] += 1;

            // max_indices：标记各segment在模型中一共有多少个value
            if (curr_indices[i] < max_indices[i])
            {
                // 更新pivot值
                pivot = i;
                res.emplace_back(*this);
            }

            // 这个步骤对于你理解pivot的作用、新PT生成的过程而言，至关重要
            curr_indices[i] -= 1;
        }
        pivot = init_pivot;
        return res;
    }

    return res;
}


// 这个函数是PCFG并行化算法的主要载体
// 尽量看懂，然后进行并行实现
// extern void gpu_generate(char **h_input, int count, const std::string &prefix, std::vector<std::string> &result);

// void PriorityQueue::Generate(PT pt)
// {
//     CalProb(pt);

//     // 获取最后一个 segment 指针
//     segment *a = nullptr;
//     int last = pt.content.size() - 1;
//     const segment &seg = pt.content[last];
//     if (seg.type == 1) a = &m.letters[m.FindLetter(seg)];
//     if (seg.type == 2) a = &m.digits[m.FindDigit(seg)];
//     if (seg.type == 3) a = &m.symbols[m.FindSymbol(seg)];

//     // 拷贝 ordered_values 到 char**
//     std::vector<std::string> values = a->ordered_values;
//     std::vector<char *> c_values(values.size());
//     for (int i = 0; i < values.size(); ++i)
//     {
//         c_values[i] = new char[64];
//         strcpy(c_values[i], values[i].c_str());
//     }

//     std::string prefix = "";
//     if (pt.content.size() > 1)
//     {
//         int seg_idx = 0;
//         for (int idx : pt.curr_indices)
//         {
//             if (pt.content[seg_idx].type == 1)
//                 prefix += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
//             if (pt.content[seg_idx].type == 2)
//                 prefix += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
//             if (pt.content[seg_idx].type == 3)
//                 prefix += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
//             seg_idx++;
//             if (seg_idx == pt.content.size() - 1)
//                 break;
//         }
//     }

//     // 使用 GPU 并行生成密码
//     std::vector<std::string> results;
//     gpu_generate(c_values.data(), values.size(), prefix, results);

//     for (auto &s : results)
//     {
//         guesses.emplace_back(s);
//         total_guesses++;
//     }

//     // 释放 host 分配的 char*
//     for (auto ptr : c_values)
//         delete[] ptr;
// }


//GPU基础
// 直接声明 gpu_generate，无需头文件
// extern "C" void gpu_generate(char *flat_input, int count, const std::string &prefix, std::vector<std::string> &result);

// void PriorityQueue::Generate(PT pt) {
//     CalProb(pt);

//     segment *a = nullptr;
//     int last = pt.content.size() - 1;
//     const segment &seg = pt.content[last];
//     if (seg.type == 1) a = &m.letters[m.FindLetter(seg)];
//     else if (seg.type == 2) a = &m.digits[m.FindDigit(seg)];
//     else if (seg.type == 3) a = &m.symbols[m.FindSymbol(seg)];
//     else {
//         std::cerr << "Unknown segment type in Generate()\n";
//         return;
//     }

//     const std::vector<std::string> &values = a->ordered_values;
//     int count = values.size();

//     std::string prefix = "";
//     if (pt.content.size() > 1) {
//         int seg_idx = 0;
//         for (int idx : pt.curr_indices) {
//             if (pt.content[seg_idx].type == 1)
//                 prefix += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
//             else if (pt.content[seg_idx].type == 2)
//                 prefix += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
//             else if (pt.content[seg_idx].type == 3)
//                 prefix += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];

//             seg_idx++;
//             if (seg_idx == pt.content.size() - 1)
//                 break;
//         }
//     }

//     const int MAX_LEN = 64;
//     int batch_size = 10000;
//     std::vector<std::string> results;  // 在循环外定义

//     for (int batch_start = 0; batch_start < count; batch_start += batch_size) {
//         int current_batch_size = std::min(batch_size, count - batch_start);

//         // 构建当前批次的扁平输入
//         std::vector<char> flat_input(current_batch_size * MAX_LEN, 0);
//         for (int i = 0; i < current_batch_size; ++i) {
//             strncpy(&flat_input[i * MAX_LEN], values[batch_start + i].c_str(), MAX_LEN - 1);
//         }

//         // 调用 GPU 并行函数
//         gpu_generate(flat_input.data(), current_batch_size, prefix, results);

//         // 将结果保存到 guesses 中
//         for (const auto &s : results) {
//             guesses.emplace_back(s);
//             total_guesses++;
//         }
//     }
// }


//进阶2


// extern "C" void gpu_generate(char *flat_input,
//                              int count,
//                              const std::string &prefix,
//                              std::vector<std::string> &result);

// void PriorityQueue::Generate(PT pt)
// {
//     CalProb(pt);

//     /* -------- 原来就有的前缀与 value 列表准备 -------- */
//     segment *a = nullptr;
//     int last = pt.content.size() - 1;
//     const segment &seg = pt.content[last];
//     if (seg.type == 1)       a = &m.letters[m.FindLetter(seg)];
//     else if (seg.type == 2)  a = &m.digits[m.FindDigit(seg)];
//     else if (seg.type == 3)  a = &m.symbols[m.FindSymbol(seg)];
//     else { std::cerr << "Unknown segment type\n"; return; }

//     const std::vector<std::string> &values = a->ordered_values;
//     int total_cnt = static_cast<int>(values.size());

//     std::string prefix;
//     if (pt.content.size() > 1) {
//         int seg_idx = 0;
//         for (int idx : pt.curr_indices) {
//             const segment &cur = pt.content[seg_idx];
//             if (cur.type == 1)
//                 prefix += m.letters[m.FindLetter(cur)].ordered_values[idx];
//             else if (cur.type == 2)
//                 prefix += m.digits[m.FindDigit(cur)].ordered_values[idx];
//             else if (cur.type == 3)
//                 prefix += m.symbols[m.FindSymbol(cur)].ordered_values[idx];
//             if (++seg_idx == pt.content.size() - 1) break;
//         }
//     }

//     /* -------- 改动从这里开始：CPU / GPU 流水线 -------- */
//     constexpr int MAX_LEN   = 64;
//     constexpr int BATCH_SZ  = 10'000;

//     std::future<std::vector<std::string>> fut;   // 保存异步 GPU 任务
//     bool pending = false;                        // 是否有尚未取回的 GPU 结果

//     auto launch_gpu = [&](std::vector<char> &&buf, int cnt)
//             -> std::future<std::vector<std::string>>
//     {
//         /* prefix 每批都一样，按值捕获即可 */
//         return std::async(std::launch::async,
//                           [pref = prefix, data = std::move(buf), cnt]() mutable
//         {
//             std::vector<std::string> res;
//             gpu_generate(data.data(), cnt, pref, res);   // GPU 计算
//             return res;                                  // 返回给 CPU
//         });
//     };

//     for (int start = 0; start < total_cnt; start += BATCH_SZ)
//     {
//         int cur_cnt = std::min(BATCH_SZ, total_cnt - start);

//         /* 1️⃣ CPU 先为“下一批”准备 flat_input --------------- */
//         std::vector<char> flat_input(cur_cnt * MAX_LEN, 0);
//         for (int i = 0; i < cur_cnt; ++i)
//             std::strncpy(&flat_input[i * MAX_LEN],
//                          values[start + i].c_str(),
//                          MAX_LEN - 1);

//         /* 2️⃣ 如果上一批 GPU 还在跑，CPU 此时正好忙完准备，
//               现在去取回结果；把“等待”时间隐藏掉 ---------- */
//         if (pending)
//         {
//             for (auto &s : fut.get()) {          // 取回上一批
//                 guesses.emplace_back(std::move(s));
//                 ++total_guesses;
//             }
//             pending = false;
//         }

//         /* 3️⃣ 把当前批扔给 GPU，立即继续循环（准备下一批） -- */
//         fut = launch_gpu(std::move(flat_input), cur_cnt);
//         pending = true;
//     }

//     /* 4️⃣ 循环结束后还有最后一批结果没取，取回来 --------- */
//     if (pending)
//     {
//         for (auto &s : fut.get()) {
//             guesses.emplace_back(std::move(s));
//             ++total_guesses;
//         }
//     }
// }

//进阶3
//按PT大小自适应：小批CPU，大批GPU+流水线
extern "C" void gpu_generate(char *flat_input,
                             int count,
                             const std::string &prefix,
                             std::vector<std::string> &result);

void PriorityQueue::Generate(PT pt)
{
    /* ---------- 1. 计算 prefix 与候选 value 列表 ---------- */
    CalProb(pt);

    segment *a = nullptr;
    const segment &last_seg = pt.content.back();
    if (last_seg.type == 1)       a = &m.letters[m.FindLetter(last_seg)];
    else if (last_seg.type == 2)  a = &m.digits[m.FindDigit(last_seg)];
    else if (last_seg.type == 3)  a = &m.symbols[m.FindSymbol(last_seg)];
    else { std::cerr << "Unknown segment type\n"; return; }

    const auto &values = a->ordered_values;
    const int  total_cnt = static_cast<int>(values.size());

    std::string prefix;
    if (pt.content.size() > 1) {
        int seg_idx = 0;
        for (int idx : pt.curr_indices) {
            const segment &cur = pt.content[seg_idx];
            if (cur.type == 1)
                prefix += m.letters[m.FindLetter(cur)].ordered_values[idx];
            else if (cur.type == 2)
                prefix += m.digits[m.FindDigit(cur)].ordered_values[idx];
            else if (cur.type == 3)
                prefix += m.symbols[m.FindSymbol(cur)].ordered_values[idx];
            if (++seg_idx == pt.content.size() - 1) break;
        }
    }

    /* ---------- 2. 按规模选择 CPU or GPU ---------- */
    constexpr int MAX_LEN       = 64;      // GPU 侧固定
    constexpr int GPU_THRESHOLD = 2'000;   // 少于此值转 CPU
    constexpr int GPU_BSZ_SMALL = 10'000;  // ≤ 50k 用 10k
    constexpr int GPU_BSZ_LARGE = 20'000;  // > 50k  用 20k

    /* ---- 2-A. 小批量：直接 CPU 拼接 ---- */
    if (total_cnt < GPU_THRESHOLD)
    {
        for (const auto &v : values) {
            guesses.emplace_back(prefix + v);
        }
        total_guesses += total_cnt;
        return;                 // 完事
    }

    /* ---- 2-B. 大批量：GPU + CPU 流水线 ---- */
    const int BATCH_SZ = (total_cnt > 50'000 ? GPU_BSZ_LARGE
                                             : GPU_BSZ_SMALL);

    std::future<std::vector<std::string>> fut;   // 尚未取回的 GPU 任务
    bool pending = false;

    auto launch_gpu = [&](std::vector<char> &&buf, int cnt)
            -> std::future<std::vector<std::string>>
    {
        return std::async(std::launch::async,
                          [pref = prefix, data = std::move(buf), cnt]() mutable
        {
            std::vector<std::string> res;
            gpu_generate(data.data(), cnt, pref, res);
            return res;
        });
    };

    for (int start = 0; start < total_cnt; start += BATCH_SZ)
    {
        int cur_cnt = std::min(BATCH_SZ, total_cnt - start);

        /* ① CPU 先准备当前批的扁平输入 ------------------ */
        std::vector<char> flat(cur_cnt * MAX_LEN, 0);
        for (int i = 0; i < cur_cnt; ++i)
            std::strncpy(&flat[i * MAX_LEN],
                         values[start + i].c_str(),
                         MAX_LEN - 1);

        /* ② 这时如果上一批 GPU 已提交，去取结果 -------- */
        if (pending)
        {
            for (auto &s : fut.get()) {
                guesses.emplace_back(std::move(s));
                ++total_guesses;
            }
            pending = false;
        }

        /* ③ 把当前批扔给 GPU，CPU 立刻回到循环准备下一批 */
        fut = launch_gpu(std::move(flat), cur_cnt);
        pending = true;
    }

    /* ④ 取回最后一批结果 ------------------------------ */
    if (pending)
    {
        for (auto &s : fut.get()) {
            guesses.emplace_back(std::move(s));
            ++total_guesses;
        }
    }
}
