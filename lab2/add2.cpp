#include<iostream>
#include<vector>
#include <windows.h>
using namespace std;

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

    int sum=0,sum1 = 0,sum2=0;
    for (int i = 0; i < n; i+=2) {
        sum1+= data[i];
        sum2 += data[i + 1];
    }
    sum=sum1+sum2;
    QueryPerformanceCounter(&end);

    double elapsedTime = static_cast<double>(end.QuadPart - start.QuadPart) / frequency.QuadPart;

    cout << "Sum: " << sum << endl;
    cout << "Execution time: " << elapsedTime << " seconds" << endl;

    return 0;
}
