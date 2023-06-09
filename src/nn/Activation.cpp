//
// Created by carlostojal on 29-05-2023.
//

#include <tojalgrad/nn/Activation.h>

namespace tojalgrad::nn {

    float Activation::linear(float in) {
        return in;
    }

    float Activation::step(float in) {
        return in < 0 ? 0 : 1;
    }

    float Activation::sigmoid(float in) {
        return 1 / (1 + std::exp(-in));
    }

    float Activation::tanh(float in) {
        return (std::exp(in) - std::exp(-in)) / (std::exp(in) + std::exp(-in));
    }

    float Activation::ReLU(float in) {
        return std::max(0.0f, in);
    }

} // nn