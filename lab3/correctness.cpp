#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <vector>
using namespace std;
using namespace chrono;

// 编译指令（需添加SIMD编译选项）：
// g++ correctness.cpp train.cpp guessing.cpp md5.cpp -o test.exe -O3 -mcpu=native -mfpu=neon

// 测试SIMD MD5的批量处理能力
void test_md5_simd() {
    const string test_inputs[4] = {  // 改为固定大小数组
        "bvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdva",
        "test123",
        "password",
        "hello_world"
    };

    bit32 simd_states[4][4];  // 改为固定大小数组
    
    MD5Hash_SIMD(test_inputs, simd_states);

    for (int i = 0; i < 4; ++i) {
        cout << "Input " << i << " (" << test_inputs[i].substr(0, 10) << "...): ";
        bit32 single_state[4];
        MD5Hash(test_inputs[i], single_state);
        
        bool match = true;
        for (int j = 0; j < 4; ++j) {
            cout << setw(8) << setfill('0') << hex << simd_states[i][j];
            if (single_state[j] != simd_states[i][j]) match = false;
        }
        cout << "  " << (match ? "[PASS]" : "[FAIL]") << endl;
    }
}

int main() {
    // 原始单线程测试
    bit32 state[4];
    MD5Hash("bvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdva", state);
    
    cout << "Single-thread MD5: ";
    for (int i = 0; i < 4; ++i) {
        cout << setw(8) << setfill('0') << hex << state[i];
    }
    cout << endl << endl;

    // 新增SIMD测试
    cout << "=== SIMD MD5 Test ===" << endl;
    test_md5_simd();

    return 0;
}
