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
}

TORCH_LIBRARY(wkv5, m) {
    m.def("forward_cpu", forward_cpu);
}


