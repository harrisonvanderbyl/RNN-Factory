#include <torch/extension.h>
#include "ATen/ATen.h"
typedef at::BFloat16 bf16;
typedef at::Half fp16;
typedef float fp32;

void cuda_forward(int B, int T, int C, int H, bf16 *r, bf16 *k, bf16 *v, float *w, bf16 *u, bf16 *y);
void cuda_backward(int B, int T, int C, int H, bf16 *r, bf16 *k, bf16 *v, float *w, float *ww, bf16 *u, bf16 *gy, bf16 *gr, bf16 *gk, bf16 *gv, bf16 *gw, bf16 *gu);
void cuda_forward_bf16(int B, int T, int C, int H, float *state, bf16 *r, bf16 *k, bf16 *v, float *w, bf16 *u, bf16 *y);
void cuda_forward_fp16(int B, int T, int C, int H, float *state, fp16 *r, fp16 *k, fp16 *v, float *w, fp16 *u, fp16 *y);
void cuda_forward_fp32(int B, int T, int C, int H, float *state, fp32 *r, fp32 *k, fp32 *v, float *w, fp32 *u, fp32 *y);



void forward_bf16(int64_t B, int64_t T, int64_t C, int64_t H, torch::Tensor &state, torch::Tensor &r, torch::Tensor &k, torch::Tensor &v, torch::Tensor &w, torch::Tensor &u, torch::Tensor &y) {
    cuda_forward_bf16(B, T, C, H, state.data_ptr<float>(), r.data_ptr<bf16>(), k.data_ptr<bf16>(), v.data_ptr<bf16>(), w.data_ptr<float>(), u.data_ptr<bf16>(), y.data_ptr<bf16>());
}
void forward_fp16(int64_t B, int64_t T, int64_t C, int64_t H, torch::Tensor &state, torch::Tensor &r, torch::Tensor &k, torch::Tensor &v, torch::Tensor &w, torch::Tensor &u, torch::Tensor &y) {
    cuda_forward_fp16(B, T, C, H, state.data_ptr<float>(), r.data_ptr<fp16>(), k.data_ptr<fp16>(), v.data_ptr<fp16>(), w.data_ptr<float>(), u.data_ptr<fp16>(), y.data_ptr<fp16>());
}
void forward_fp32(int64_t B, int64_t T, int64_t C, int64_t H, torch::Tensor &state, torch::Tensor &r, torch::Tensor &k, torch::Tensor &v, torch::Tensor &w, torch::Tensor &u, torch::Tensor &y) {
    cuda_forward_fp32(B, T, C, H, state.data_ptr<float>(), r.data_ptr<fp32>(), k.data_ptr<fp32>(), v.data_ptr<fp32>(), w.data_ptr<float>(), u.data_ptr<fp32>(), y.data_ptr<fp32>());
}


void forward(int64_t B, int64_t T, int64_t C, int64_t H, torch::Tensor &r, torch::Tensor &k, torch::Tensor &v, torch::Tensor &w, torch::Tensor &u, torch::Tensor &y) {
    cuda_forward(B, T, C, H, r.data_ptr<bf16>(), k.data_ptr<bf16>(), v.data_ptr<bf16>(), w.data_ptr<float>(), u.data_ptr<bf16>(), y.data_ptr<bf16>());
}
void backward(int64_t B, int64_t T, int64_t C, int64_t H, torch::Tensor &r, torch::Tensor &k, torch::Tensor &v, torch::Tensor &w, torch::Tensor &ww, torch::Tensor &u, torch::Tensor &gy, torch::Tensor &gr, torch::Tensor &gk, torch::Tensor &gv, torch::Tensor &gw, torch::Tensor &gu) {
    cuda_backward(B, T, C, H, r.data_ptr<bf16>(), k.data_ptr<bf16>(), v.data_ptr<bf16>(), w.data_ptr<float>(), ww.data_ptr<float>(), u.data_ptr<bf16>(), gy.data_ptr<bf16>(), gr.data_ptr<bf16>(), gk.data_ptr<bf16>(), gv.data_ptr<bf16>(), gw.data_ptr<bf16>(), gu.data_ptr<bf16>());
}

// simd
#include <immintrin.h>

#ifdef __AVX512F__  // This macro is defined if AVX-512 is supported
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
  #define SIMD_WIDTH 8
  #define LOAD(x) _mm256_load_ps(x)
  #define STORE(x, y) _mm256_store_ps(x, y)
  #define SET1(x) _mm256_set1_ps(x)
  #define MULTIPLY(x, y) _mm256_mul_ps(x, y)
  #define MULTADD(x, y, z) _mm256_fmadd_ps(x, y, z)
  // print out the SIMD width
    #pragma message("AVX-512 is not supported")
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
    m.def("forward", &forward, "wkv5 forward");
    m.def("forward_cpu", &forward_cpu, "CPU forward");
    m.def("backward", &backward, "wkv5 backward");
    m.def("forward_bf16", &forward_bf16, "rwkv5 forward_bf16");
    m.def("forward_fp16", &forward_fp16, "rwkv5 forward_fp16");
    m.def("forward_fp32", &forward_fp32, "rwkv5 forward_fp32");
}

TORCH_LIBRARY(wkv5, m) {
    m.def("forward", forward);
    m.def("forward_cpu", forward_cpu);
    m.def("backward", backward);
    m.def("forward_bf16", forward_bf16);
    m.def("forward_fp16", forward_fp16);
    m.def("forward_fp32", forward_fp32);
}


