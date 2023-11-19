#include <torch/extension.h>
#include "ATen/ATen.h"
typedef at::BFloat16 bf16;
typedef at::Half fp16;
typedef float fp32;

// simd
#ifdef __AVX512F__  // This macro is defined if AVX-512 is supported
  #include <immintrin.h>
  
  #define SIMD_WIDTH 16
  #define LOAD(x) _mm512_load_ps(x)
  #define STORE(x, y) _mm512_store_ps(x, y)
  #define SET1(x) _mm512_set1_ps(x)
  #define MULTIPLY(x, y) _mm512_mul_ps(x, y)
  #define MULTADD(x, y, z) _mm512_fmadd_ps(x, y, z)
  // print out the SIMD width
    #pragma message("AVX-512 is supported")
#else
  // Fallback to AVX2 if AVX-512 is not supported
  #ifdef __AVX2__
    #include <immintrin.h>
    #define SIMD_WIDTH 8
    #define LOAD(x) _mm256_load_ps(x)
    #define STORE(x, y) _mm256_store_ps(x, y)
    #define SET1(x) _mm256_set1_ps(x)
    #define MULTIPLY(x, y) _mm256_mul_ps(x, y)
    #define MULTADD(x, y, z) _mm256_fmadd_ps(x, y, z)
    // print out the SIMD width
        #pragma message("AVX-512 is not supported")

  #else
    #if defined(__ARM_NEON) || defined(__ARM_NEON__)
        #include <arm_neon.h>
        #define SIMD_WIDTH 4  // NEON typically operates on 128-bit registers (4 floats)
        #define LOAD(x) vld1q_f32(x)
        #define STORE(x, y) vst1q_f32(x, y)
        #define SET1(x) vdupq_n_f32(x)
        #define MULTIPLY(x, y) vmulq_f32(x, y)
        #define MULTADD(x, y, z) vmlaq_f32(z, x, y)
        // Print out the SIMD width
        #pragma message("ARM NEON is supported")
    #else
        #pragma message("No SIMD is supported")
        #define SIMD_WIDTH 1
        #define LOAD(x) x
        #define STORE(x, y) x = y
        #define SET1(x) x
        #define MULTIPLY(x, y) x * y
        #define MULTADD(x, y, z) x * y + z
    #endif

    #endif

#endif

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

    const long inblocksize = 128;
    const long horizontal_block_size = 16;

    // Parallel computation
    #pragma omp parallel for collapse(3) shared(A, Ar, Ao, B, C)
    for (long i = 0; i < OUT; i += 1) {
        for (long kk = 0; kk < IN; kk += inblocksize){
            for (long bbb = 0; bbb < BB; bbb += horizontal_block_size){
            long io = i << 4;
            __m512 Ario = load_aligned(&Ar[io]);
            __m512 Aoio = load_aligned(&Ao[io]);
            for (long k = kk; k < kk + inblocksize; k += 16) {
                __m512 aa = convert_uint8_to_ps(&A[i * IN + k]);
                for (long j = 0; j < T; j += 1) {
                    for (long bb = bbb; bb < std::min(bbb+horizontal_block_size,BB); bb += 1) {
                        
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




void forward_cpu(int64_t B, int64_t T, int64_t C, int64_t H, torch::Tensor &s, torch::Tensor &r, torch::Tensor &k, torch::Tensor &v, torch::Tensor &w, torch::Tensor &u, torch::Tensor &y) {
    
    auto rr = r.data_ptr<float>();
    auto kk = k.data_ptr<float>();
    auto vv = v.data_ptr<float>();
    auto ww = w.data_ptr<float>();
    auto uu = u.data_ptr<float>();
    auto ss = s.data_ptr<float>();
    auto out = y.data_ptr<float>();

    // 1d tensor
    int64_t tsize = B*H*(C/H);
    // 2d tensor
    int64_t ttsize = B*H*(C/H)*(C/H);

    // 1d 
    int64_t bsize = H*(C/H);
    // 2d
    int64_t bbsize = H*(C/H)*(C/H);

    // 1d
    int64_t hsize = (C/H);
    // 2d
    int64_t hhsize = (C/H)*(C/H);

    for (int64_t t = 0; t < T; t++) {

        auto timeoffset = t * tsize;

        for (int64_t bb = 0; bb < B; bb++) {

            auto btimeoffset = timeoffset + bb * bsize;
            auto bbhsize = bb * bbsize;

            for (int64_t hh = 0; hh < H; hh++) {
                auto hoffset = hh * hsize;
                auto bhofseti = btimeoffset + hoffset;
                auto bbhhsize = bbhsize + hh * hhsize;

                for (int64_t i = 0; i < C/H; i++) {

                    int64_t iind = bhofseti + i;
                    auto hoffseti = hoffset + i;  
                    auto bbhhofseti = bbhhsize + i * hsize;  

                    //auto kkk = kk[iind];
                    auto kkk = SET1(kk[iind]);  
                    auto uuu = SET1(uu[hoffseti]); 
                    auto rrr = SET1(rr[iind]);
                    auto www = SET1(ww[hoffseti]);

                    for (int64_t j = 0; j < C/H; j+=SIMD_WIDTH) {
                        int64_t jind = bhofseti + j;
                        int64_t sind = bbhhofseti + j;

                        
                        
                        // atu = k[t,bb,hh,i]*v[t,bb,hh,j]
                        auto vvv = LOAD(&vv[jind]);

                        // multiply kkk and vvv
                        auto atu = MULTIPLY(vvv,kkk);

                        auto sss = LOAD(&ss[sind]);

                        // out[t,bb,hh,j] += r[t,bb,hh,i]*(s[bb,hh,i,j] + atu*u[hh,i] )
                        auto sssatuuuu = MULTADD(atu,uuu,sss);

                        auto outtt = LOAD(&out[jind]);

                        STORE(&out[jind], MULTADD(sssatuuuu,rrr,outtt));

                        // s[bb,hh,i,j] = s[bb,hh,i,j] * w[hh,i] + atu
                        STORE(&ss[sind], MULTADD(sss,www,atu));
                    }

                }
            }
        }
    }
          
}




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_cpu", &forward_cpu, "CPU forward");
    m.def("quantize_cpu", &Quantize, "QuantizeCpu");
    m.def("matmul", &matmul_avx512_optimized, "matmul_avx512I");
    
}

TORCH_LIBRARY(wkv5, m) {
    m.def("forward_cpu", forward_cpu);
    m.def("quantize_cpu", Quantize);
    m.def("matmul", matmul_avx512_optimized);
}


