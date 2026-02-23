#pragma once
// ============================================================================
// AgentKV SIMD Distance Kernels
// Compile-time selection: AVX2 > SSE2 > scalar fallback
// ============================================================================
#include <cstdint>
#include <cmath>

// Detect x86 architecture
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define AGENTKV_X86 1
#include <immintrin.h>
#endif

namespace simd {

// ─────────────────────────────────────────────────────────────────────────────
// Scalar fallback (auto-vectorized by compiler with -O3 / /O2)
// ─────────────────────────────────────────────────────────────────────────────
inline float dot_scalar(const float* a, const float* b, uint32_t dim) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < dim; ++i) sum += a[i] * b[i];
    return sum;
}

inline float l2sq_scalar(const float* a, const float* b, uint32_t dim) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < dim; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

#ifdef AGENTKV_X86

// ─────────────────────────────────────────────────────────────────────────────
// SSE2 kernels (always available on x86-64)
// Process 4 floats per iteration
// ─────────────────────────────────────────────────────────────────────────────
inline float dot_sse2(const float* a, const float* b, uint32_t dim) {
    __m128 sum0 = _mm_setzero_ps();
    __m128 sum1 = _mm_setzero_ps();
    uint32_t i = 0;
    // Unrolled 2x for better ILP (8 floats per iteration)
    for (; i + 8 <= dim; i += 8) {
        sum0 = _mm_add_ps(sum0, _mm_mul_ps(_mm_loadu_ps(a + i),     _mm_loadu_ps(b + i)));
        sum1 = _mm_add_ps(sum1, _mm_mul_ps(_mm_loadu_ps(a + i + 4), _mm_loadu_ps(b + i + 4)));
    }
    for (; i + 4 <= dim; i += 4) {
        sum0 = _mm_add_ps(sum0, _mm_mul_ps(_mm_loadu_ps(a + i), _mm_loadu_ps(b + i)));
    }
    sum0 = _mm_add_ps(sum0, sum1);
    // Horizontal sum: sum all 4 lanes
    __m128 shuf = _mm_shuffle_ps(sum0, sum0, _MM_SHUFFLE(2, 3, 0, 1));
    sum0 = _mm_add_ps(sum0, shuf);
    shuf = _mm_movehl_ps(shuf, sum0);
    sum0 = _mm_add_ss(sum0, shuf);
    float result = _mm_cvtss_f32(sum0);
    for (; i < dim; i++) result += a[i] * b[i];
    return result;
}

inline float l2sq_sse2(const float* a, const float* b, uint32_t dim) {
    __m128 sum0 = _mm_setzero_ps();
    __m128 sum1 = _mm_setzero_ps();
    uint32_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m128 d0 = _mm_sub_ps(_mm_loadu_ps(a + i),     _mm_loadu_ps(b + i));
        __m128 d1 = _mm_sub_ps(_mm_loadu_ps(a + i + 4), _mm_loadu_ps(b + i + 4));
        sum0 = _mm_add_ps(sum0, _mm_mul_ps(d0, d0));
        sum1 = _mm_add_ps(sum1, _mm_mul_ps(d1, d1));
    }
    for (; i + 4 <= dim; i += 4) {
        __m128 d = _mm_sub_ps(_mm_loadu_ps(a + i), _mm_loadu_ps(b + i));
        sum0 = _mm_add_ps(sum0, _mm_mul_ps(d, d));
    }
    sum0 = _mm_add_ps(sum0, sum1);
    __m128 shuf = _mm_shuffle_ps(sum0, sum0, _MM_SHUFFLE(2, 3, 0, 1));
    sum0 = _mm_add_ps(sum0, shuf);
    shuf = _mm_movehl_ps(shuf, sum0);
    sum0 = _mm_add_ss(sum0, shuf);
    float result = _mm_cvtss_f32(sum0);
    for (; i < dim; i++) { float d = a[i] - b[i]; result += d * d; }
    return result;
}

#ifdef __AVX2__
// ─────────────────────────────────────────────────────────────────────────────
// AVX2 + FMA kernels (requires -mavx2 -mfma or /arch:AVX2)
// Process 8 floats per iteration with fused multiply-add
// ─────────────────────────────────────────────────────────────────────────────
inline float dot_avx2(const float* a, const float* b, uint32_t dim) {
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    uint32_t i = 0;
    // Unrolled 2x (16 floats per iteration)
    for (; i + 16 <= dim; i += 16) {
        sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),     _mm256_loadu_ps(b + i),     sum0);
        sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), sum1);
    }
    for (; i + 8 <= dim; i += 8) {
        sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), sum0);
    }
    sum0 = _mm256_add_ps(sum0, sum1);
    // Horizontal sum: 256-bit -> 128-bit -> scalar
    __m128 hi = _mm256_extractf128_ps(sum0, 1);
    __m128 lo = _mm256_castps256_ps128(sum0);
    __m128 s = _mm_add_ps(hi, lo);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    float result = _mm_cvtss_f32(s);
    for (; i < dim; i++) result += a[i] * b[i];
    return result;
}

inline float l2sq_avx2(const float* a, const float* b, uint32_t dim) {
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    uint32_t i = 0;
    for (; i + 16 <= dim; i += 16) {
        __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a + i),     _mm256_loadu_ps(b + i));
        __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8));
        sum0 = _mm256_fmadd_ps(d0, d0, sum0);
        sum1 = _mm256_fmadd_ps(d1, d1, sum1);
    }
    for (; i + 8 <= dim; i += 8) {
        __m256 d = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        sum0 = _mm256_fmadd_ps(d, d, sum0);
    }
    sum0 = _mm256_add_ps(sum0, sum1);
    __m128 hi = _mm256_extractf128_ps(sum0, 1);
    __m128 lo = _mm256_castps256_ps128(sum0);
    __m128 s = _mm_add_ps(hi, lo);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    float result = _mm_cvtss_f32(s);
    for (; i < dim; i++) { float d = a[i] - b[i]; result += d * d; }
    return result;
}
#endif // __AVX2__

// ─────────────────────────────────────────────────────────────────────────────
// Dispatchers — pick the best available kernel at compile time
// ─────────────────────────────────────────────────────────────────────────────
inline float dot(const float* a, const float* b, uint32_t dim) {
#ifdef __AVX2__
    return dot_avx2(a, b, dim);
#else
    return dot_sse2(a, b, dim);
#endif
}

inline float l2sq(const float* a, const float* b, uint32_t dim) {
#ifdef __AVX2__
    return l2sq_avx2(a, b, dim);
#else
    return l2sq_sse2(a, b, dim);
#endif
}

#else // !AGENTKV_X86  (ARM, RISC-V, etc.)

inline float dot(const float* a, const float* b, uint32_t dim) {
    return dot_scalar(a, b, dim);
}

inline float l2sq(const float* a, const float* b, uint32_t dim) {
    return l2sq_scalar(a, b, dim);
}

#endif // AGENTKV_X86

} // namespace simd
