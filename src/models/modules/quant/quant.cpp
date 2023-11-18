#include <torch/extension.h>
#include "ATen/ATen.h"
#include <immintrin.h>

void matmul_avx512I(torch::Tensor &At, torch::Tensor &Art, torch::Tensor &Aot,torch::Tensor &Bt, torch::Tensor &Ct,long BB, long M, long N, long K) {
    u_char* A = At.data_ptr<u_char>();
    float* Ar = Art.data_ptr<float>();
    float* Ao = Aot.data_ptr<float>();
    float* B = Bt.data_ptr<float>();
    float* C = Ct.data_ptr<float>();
    long i, j, k, bb;
    __m512 a, b, c, d;
    for (i = 0; i < M; i++) {
        for (k = 0; k < K; k++) {
            a = _mm512_set1_ps(float(A[i * K + k]) * Ar[i * 16 + k%16] + Ao[i * 16 + k%16]);
            for ( bb = 0; bb < BB; bb += 1){
            
                for (j = 0; j < N; j += 16) {
                    b = _mm512_loadu_ps(B + bb*N*K + k * N + j);
                    c = _mm512_loadu_ps(C + bb*N*K + i * N + j);
                    d = _mm512_fmadd_ps(a, b, c);
                    _mm512_storeu_ps(C + bb*N*K + i * N + j, d);
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
            __m512 a = _mm512_loadu_ps(A + i * N + j);
            max = _mm512_max_ps(max, a);
            min = _mm512_min_ps(min, a);
        }
        __m512 range = _mm512_sub_ps(max, min);
        __m512 scale = _mm512_div_ps( range, _mm512_set1_ps(255));
        _mm512_storeu_ps(Ar + i * 16, scale);
        _mm512_storeu_ps(Ao + i * 16, min);
        for (j = 0; j < N; j += 16) {
            __m512 a = _mm512_loadu_ps(A + i * N + j);
            
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