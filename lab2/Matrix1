#include<iostream>
#include<vector>
#include <windows.h>
using namespace std;

void calculateMatrixProduct(const vector<vector<double>>& matrix, const vector<double>& vec, vector<double>& result)
{
	int n = matrix.size();
	for (int i = 0; i < n; i++)
	{
		result[i] = 0.0;
		for (int j = 0; j < n; j++)
		{
			result[i] += matrix[j][i] * vec[j];
		}
	}
}

int main()
{
	int n;
	cin >> n;
	vector<vector<double>> matrix(n, vector<double>(n));
	vector<double> vec(n), result(n);
	srand(static_cast<unsigned>(time(nullptr)));
	for (int i = 0; i < n; ++i) {
		vec[i] = i;
		for (int j = 0; j < n; ++j) {
			matrix[i][j] = i + j;
		}
	}
	LARGE_INTEGER frequency, start, end;
	QueryPerformanceFrequency(&frequency);
	QueryPerformanceCounter(&start);

	calculateMatrixProduct(matrix, vec, result);

	QueryPerformanceCounter(&end);

	double elapsedTime = static_cast<double>(end.QuadPart - start.QuadPart) / frequency.QuadPart;

	cout << "Execution time: " << elapsedTime << " seconds" << endl;
	return 0;
}
