#include <iostream>
#include <vector>
#include <windows.h>
#include <omp.h>
using namespace std;
//使用 OpenMP 让多个 核心（core） 并行计算部分和，最后再归约。

double parallel_sum(const vector<double>& data) {
    double sum = 0.0;
#pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < data.size(); ++i) {
        sum += data[i];
    }
    return sum;
}

int main() {
    int n;
    cin >> n;

    vector<double> data(n);
    for (int i = 0; i < n; i++) {
        data[i] = i + 1;
    }

    LARGE_INTEGER frequency, start, end;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);

    double sum = parallel_sum(data);
    QueryPerformanceCounter(&end);

    double elapsedTime = static_cast<double>(end.QuadPart - start.QuadPart) / frequency.QuadPart;

    cout << "Sum: " << sum << endl;
    cout << "Execution time: " << elapsedTime << " seconds" << endl;

    return 0;
}
