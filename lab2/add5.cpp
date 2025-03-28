#include <iostream>
#include <vector>
#include <windows.h>  // 高精度计时

using namespace std;

// 4路循环展开求和
double unrolled_sum(const vector<double>& data) {
    double sum = 0.0;
    size_t i = 0;
    const size_t n = data.size();

    // 主循环：每次处理4个元素
    for (; i + 4 <= n; i += 4) {
        sum += data[i] + data[i + 1]
            + data[i + 2] + data[i + 3];
    }

    // 处理剩余元素（不足4个时）
    for (; i < n; i++) {
        sum += data[i];
    }
    return sum;
}

int main() {
    int n;
    cout << "Enter number of elements: ";
    cin >> n;

    // 初始化数据（1到n的连续值，方便验证）
    vector<double> data(n);
    for (int i = 0; i < n; i++) {
        data[i] = i + 1;
    }

    // 计时开始
    LARGE_INTEGER frequency, start, end;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);

    // 计算求和
    double sum = unrolled_sum(data);

    // 计时结束
    QueryPerformanceCounter(&end);
    double elapsedTime = static_cast<double>(end.QuadPart - start.QuadPart) / frequency.QuadPart;

    // 输出结果
    cout << "Sum: " << sum << endl;
    cout << "Execution time: " << elapsedTime << " seconds" << endl;

    // 验证结果（公式：1+2+...+n = n(n+1)/2）
    double expected = n * (n + 1) / 2.0;
    if (abs(sum - expected) > 1e-6) {
        cerr << "Error: Result mismatch! Expected " << expected << ", got " << sum << endl;
        return 1;
    }

    return 0;
}