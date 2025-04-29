#ifndef MD5_H
#define MD5_H

#include <iostream>
#include <string>
#include <cstring>
#include <arm_neon.h>

using namespace std;

typedef unsigned char Byte;
typedef unsigned int bit32;

// 移位常量
#define s11 7
#define s12 12
#define s13 17
#define s14 22
#define s21 5
#define s22 9
#define s23 14
#define s24 20
#define s31 4
#define s32 11
#define s33 16
#define s34 23
#define s41 6
#define s42 10
#define s43 15
#define s44 21

// 普通 MD5 函数
#define F(x, y, z) (((x) & (y)) | ((~x) & (z)))
#define G(x, y, z) (((x) & (z)) | ((y) & (~z)))
#define H(x, y, z) ((x) ^ (y) ^ (z))
#define I(x, y, z) ((y) ^ ((x) | (~z)))

#define ROTATELEFT(num, n) (((num) << (n)) | ((num) >> (32-(n))))

#define FF(a, b, c, d, x, s, ac) { \
  (a) += F ((b), (c), (d)) + (x) + ac; \
  (a) = ROTATELEFT ((a), (s)); \
  (a) += (b); \
}

#define GG(a, b, c, d, x, s, ac) { \
  (a) += G ((b), (c), (d)) + (x) + ac; \
  (a) = ROTATELEFT ((a), (s)); \
  (a) += (b); \
}

#define HH(a, b, c, d, x, s, ac) { \
  (a) += H ((b), (c), (d)) + (x) + ac; \
  (a) = ROTATELEFT ((a), (s)); \
  (a) += (b); \
}

#define II(a, b, c, d, x, s, ac) { \
  (a) += I ((b), (c), (d)) + (x) + ac; \
  (a) = ROTATELEFT ((a), (s)); \
  (a) += (b); \
}

// SIMD RotateLeft（立即数版本）
#define ROTATELEFT_SIMD_FIXED(a, s) \
    vorrq_u32(vshlq_n_u32((a), s), vshrq_n_u32((a), 32 - s))

// SIMD 宏（使用立即数移位，编译器完全支持）
#define FF_SIMD(a, b, c, d, x, s, ac) { \
    uint32x4_t f = vorrq_u32(vandq_u32(b, c), vandq_u32(vmvnq_u32(b), d)); \
    a = vaddq_u32(a, f); \
    a = vaddq_u32(a, x); \
    a = vaddq_u32(a, vdupq_n_u32(ac)); \
    a = ROTATELEFT_SIMD_FIXED(a, s); \
    a = vaddq_u32(a, b); \
}

#define GG_SIMD(a, b, c, d, x, s, ac) { \
    uint32x4_t g = vorrq_u32(vandq_u32(b, d), vandq_u32(c, vmvnq_u32(d))); \
    a = vaddq_u32(a, g); \
    a = vaddq_u32(a, x); \
    a = vaddq_u32(a, vdupq_n_u32(ac)); \
    a = ROTATELEFT_SIMD_FIXED(a, s); \
    a = vaddq_u32(a, b); \
}

#define HH_SIMD(a, b, c, d, x, s, ac) { \
    uint32x4_t h = veorq_u32(veorq_u32(b, c), d); \
    a = vaddq_u32(a, h); \
    a = vaddq_u32(a, x); \
    a = vaddq_u32(a, vdupq_n_u32(ac)); \
    a = ROTATELEFT_SIMD_FIXED(a, s); \
    a = vaddq_u32(a, b); \
}

#define II_SIMD(a, b, c, d, x, s, ac) { \
    uint32x4_t i = veorq_u32(c, vornq_u32(b, d)); \
    a = vaddq_u32(a, i); \
    a = vaddq_u32(a, x); \
    a = vaddq_u32(a, vdupq_n_u32(ac)); \
    a = ROTATELEFT_SIMD_FIXED(a, s); \
    a = vaddq_u32(a, b); \
}

// 接口声明
void MD5Hash(string input, bit32 *state);
void MD5Hash_SIMD(const string inputs[4], bit32 states[4][4]);

#endif
