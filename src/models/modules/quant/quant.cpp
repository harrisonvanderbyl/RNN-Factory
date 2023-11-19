#include <torch/extension.h>
#include "ATen/ATen.h"
#include <immintrin.h>
#include <algorithm>
#include <omp.h>
// Helper function: Load 64 bytes with 16 floats, aligned to 64 byte boundary.
inline __m512 load_aligned(const float* ptr) {
    return _mm512_load_ps(ptr);
}

__m512 convert_uint8_to_ps(const u_char* data) {
    // Load the uint8 data into a 128-bit register as integer values
    __m128i input = _mm_loadu_si128((const __m128i*)data);

    // Zero extend 8-bit integers to 32-bit integers and then to single-precision floats
    // AVX-512 provides the _mm512_cvtepu8_epi32 which directly extends uint8 values to 32-bit integers
    __m512i extended = _mm512_cvtepu8_epi32(input);

    // Convert extended 32-bit integers to single-precision floats
    __m512 result = _mm512_cvtepi32_ps(extended);

    return result;
}
// Helper function: Load 64 bytes with 16 floats, potentially unaligned.
inline __m512 load_unaligned(const float* ptr) {
    return _mm512_loadu_ps(ptr);
}

// Helper function: Store 64 bytes with 16 floats, aligned to 64 byte boundary.
inline void store_aligned(float* ptr, __m512 value) {
    _mm512_store_ps(ptr, value);
}

void matmul_avx512_optimized(const torch::Tensor &At, const torch::Tensor &Art, const torch::Tensor &Aot,
                             const torch::Tensor &Bt, torch::Tensor &Ct,
                             const long BB, const long IN, const long T, const long OUT) {
    // Pointers to the data
    auto A = At.data_ptr<unsigned char>();
    auto Ar = Art.data_ptr<float>();
    auto Ao = Aot.data_ptr<float>();
    auto B = Bt.data_ptr<float>();
    auto C = Ct.data_ptr<float>();

    for (long i = 0; i < OUT; i += 1) {
        long io = i << 4;
        __m512 Ario = load_aligned(&Ar[io]);
        __m512 Aoio = load_aligned(&Ao[io]);
        for (long k = 0; k < IN; k += 16) {
            __m512 aa = convert_uint8_to_ps(&A[i * IN + k]);
            for (long j = 0; j < T; j += 1) {
                for (long bb = 0; bb < BB; bb += 1) {
                    
                    __m512 a = _mm512_fmadd_ps(Ario, aa, Aoio);
                         
                    __m512 b = load_aligned(&B[bb * IN * T + j * IN + k]);
                    __m512 c = _mm512_mul_ps(a, b);

                    // atomic_add(&C[bb * T * OUT + i * T + j], c);

                    C[bb * T * OUT + j * OUT + i] += c[0] + c[1] + c[2] + c[3] + c[4] + c[5] + c[6] + c[7] + c[8] + c[9] + c[10] + c[11] + c[12] + c[13] + c[14] + c[15];
                }
            }
        }
    }
}
void Quantize(torch::Tensor &At, torch::Tensor &Art, torch::Tensor &Aot, torch::Tensor &Aqt, long M, long N) {
    float* A = At.data_ptr<float>();
    float* Ar = Art.data_ptr<float>();
    float* Ao = Aot.data_ptr<float>();
    u_char* Aq = Aqt.data_ptr<u_char>();

    long i, j;
    for (i = 0; i < M; i++) {
        __m512 max = _mm512_set1_ps(-1e9);
        __m512 min = _mm512_set1_ps(1e9);
        for (j = 0; j < N; j += 16) {
            __m512 a = _mm512_load_ps(A + i * N + j);
            max = _mm512_max_ps(max, a);
            min = _mm512_min_ps(min, a);
        }
        __m512 range = _mm512_sub_ps(max, min);
        __m512 scale = _mm512_div_ps( range, _mm512_set1_ps(255));
        _mm512_store_ps(Ar + i * 16, scale);
        _mm512_store_ps(Ao + i * 16, min);
        for (j = 0; j < N; j += 16) {
            __m512 a = _mm512_load_ps(A + i * N + j);
            
            __m512 d = _mm512_div_ps(_mm512_sub_ps(a, min), scale);
            
            for (long k = 0; k < 16; k++) {
                Aq[i * N + j + k] = (u_char)long(d[k]);
            }
        }
    }
}

// pytorch bindings




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize_cpu", &Quantize, "QuantizeCpu");
    m.def("matmul", &matmul_avx512_optimized, "matmul_avx512I");
    
}

TORCH_LIBRARY(wkv5, m) {
    m.def("quantize_cpu", Quantize);
    m.def("matmul", matmul_avx512_optimized);
}