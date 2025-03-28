#include <iostream>
#include <vector>
#include <windows.h>  // �߾��ȼ�ʱ

using namespace std;

// 4·ѭ��չ�����
double unrolled_sum(const vector<double>& data) {
    double sum = 0.0;
    size_t i = 0;
    const size_t n = data.size();

    // ��ѭ����ÿ�δ���4��Ԫ��
    for (; i + 4 <= n; i += 4) {
        sum += data[i] + data[i + 1]
            + data[i + 2] + data[i + 3];
    }

    // ����ʣ��Ԫ�أ�����4��ʱ��
    for (; i < n; i++) {
        sum += data[i];
    }
    return sum;
}

int main() {
    int n;
    cout << "Enter number of elements: ";
    cin >> n;

    // ��ʼ�����ݣ�1��n������ֵ��������֤��
    vector<double> data(n);
    for (int i = 0; i < n; i++) {
        data[i] = i + 1;
    }

    // ��ʱ��ʼ
    LARGE_INTEGER frequency, start, end;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);

    // �������
    double sum = unrolled_sum(data);

    // ��ʱ����
    QueryPerformanceCounter(&end);
    double elapsedTime = static_cast<double>(end.QuadPart - start.QuadPart) / frequency.QuadPart;

    // ������
    cout << "Sum: " << sum << endl;
    cout << "Execution time: " << elapsedTime << " seconds" << endl;

    // ��֤�������ʽ��1+2+...+n = n(n+1)/2��
    double expected = n * (n + 1) / 2.0;
    if (abs(sum - expected) > 1e-6) {
        cerr << "Error: Result mismatch! Expected " << expected << ", got " << sum << endl;
        return 1;
    }

    return 0;
}