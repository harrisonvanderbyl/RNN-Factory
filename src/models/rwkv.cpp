
#include "ATen/ATen.h"
#include <torch/extension.h>
#include "rwkv.h"
RWKV* rwkv = nullptr;


void forward_cpu(torch::Tensor& inps, torch::Tensor& out) {
    
   // get unsigned ints from tensor
   size_t n = inps.size(0);
    size_t d = inps.size(1);
    std::vector<std::vector<size_t>> vecs(n, std::vector<size_t>(d));
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < d; j++) {
            vecs[i][j] = inps[i][j].item().toInt();
        }
    }
   auto oout = rwkv->operator()(vecs); 
    for (size_t i = 0; i < n; i++) {
         for (size_t j = 0; j < d; j++) {
              for (size_t k = 0; k < pow(2,16); k++) {
                  out[i][j][k] = oout[i][j][k].get<float>(0);
              }
         }
    }
}

void resetState() {
    rwkv->set_state(rwkv->new_state());
}

void init(std::string model_path) {
    rwkv = new RWKV(model_path);
}




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_cpu", &forward_cpu, "CPU forward");
    m.def("init", &init, "Initialize model");
    m.def("resetState", &resetState, "Reset model state");
}

TORCH_LIBRARY(wkv5, m) {
    m.def("forward_cpu", forward_cpu);
    m.def("init", init);
    m.def("resetState", resetState);
}


