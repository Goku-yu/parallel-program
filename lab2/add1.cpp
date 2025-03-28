#include<iostream>
#include<vector>
#include <windows.h>
using namespace std;

int main() {
	int n;
	cin >> n;
	vector<int> data(n);
	for (int i = 0; i < n; i++) {
		data[i]=i+1;
	}
    LARGE_INTEGER frequency, start, end;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);

    int sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += data[i];
    }

    QueryPerformanceCounter(&end);

    double elapsedTime = static_cast<double>(end.QuadPart - start.QuadPart) / frequency.QuadPart;

    cout << "Sum: " << sum << endl;
    cout << "Execution time: " << elapsedTime << " seconds" << endl;

    return 0;
}
