#include <iostream>
#include <vector>
#include <windows.h>
using namespace std;

void recursion(vector<int>& data, int n) {
    if (n == 1)
        return;
    else {
        for (int i = 0; i < n / 2; i++)
            data[i] += data[n - i - 1];
        n = n / 2;
        recursion(data, n);
    }
}

int main() {
    int n;
    cin >> n;

    vector<int> data(n);
    for (int i = 0; i < n; i++) {
        data[i] = i + 1;
    }

    LARGE_INTEGER frequency, start, end;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);

    recursion(data, n);
    QueryPerformanceCounter(&end);

    double elapsedTime = static_cast<double>(end.QuadPart - start.QuadPart) / frequency.QuadPart;

    cout << "Sum: " << data[0] << endl;
    cout << "Execution time: " << elapsedTime << " seconds" << endl;

    return 0;
}
