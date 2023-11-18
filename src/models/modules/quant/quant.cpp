#include <torch/extension.h>
#include "ATen/ATen.h"
#include <immintrin.h>
#include <algorithm>

void matmul_avx512I(torch::Tensor &At, torch::Tensor &Art, torch::Tensor &Aot, torch::Tensor &Bt, torch::Tensor &Ct, long BB, long M, long N, long K) {
    auto A = At.data_ptr<u_char>();
    auto Ar = Art.data_ptr<float>();
    auto Ao = Aot.data_ptr<float>();
    auto B = Bt.data_ptr<float>();
    auto C = Ct.data_ptr<float>();

    // Cache-blocking sizes, for example purposes. It should be tuned according to your CPU's cache sizes.
    const long block_size_m = 64;
    const long block_size_k = 256;
    const long block_size_n = 64;

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (long bb = 0; bb < BB; bb++) {
        for (long mj0 = 0; mj0 < M; mj0 += block_size_m) {
            for (long nj0 = 0; nj0 < N; nj0 += block_size_n) {
                for (long j = nj0; j < std::min(nj0 + block_size_n, N); j += 16) {
                    for (long i = mj0; i < std::min(mj0 + block_size_m, M); i++) {
                        // Load the elements of C once, work on a block and store it back
                        __m512 c_block[block_size_n / 16];
                        for (long jj = j; jj < j + 16; jj += 16) {
                            c_block[(jj - j) / 16] = _mm512_load_ps(&C[bb * N * M + i * N + jj]);
                        }

                        for (long k0 = 0; k0 < K; k0 += block_size_k) {
                            // Prepare blocks for A and B in the registers
                            for (long k = k0; k < std::min(k0 + block_size_k, K); k++) {
                                // Load and broadcast values of A and precomputed arrays
                                long precomp_index = k % 16;
                                __m512 a = _mm512_set1_ps(float(A[i * K + k]) * Ar[i * 16 + precomp_index] + Ao[i * 16 + precomp_index]);

                                for (long jj = j; jj < j + 16; jj += 16) {
                                    __m512 b = _mm512_load_ps(&B[bb * K * N + k * N + jj]);
                                    c_block[(jj - j) / 16] = _mm512_fmadd_ps(a, b, c_block[(jj - j) / 16]);
                                }
                            }
                        }

                        // Store the results back to the memory
                        for (long jj = j; jj < j + 16; jj += 16) {
                            _mm512_store_ps(&C[bb * N * M + i * N + jj], c_block[(jj - j) / 16]);
                        }
                    }
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
    m.def("matmul", &matmul_avx512I, "matmul_avx512I");
    
}

TORCH_LIBRARY(wkv5, m) {
    m.def("quantize_cpu", Quantize);
    m.def("matmul", matmul_avx512I);
}