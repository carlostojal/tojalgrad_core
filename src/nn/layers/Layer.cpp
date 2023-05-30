//
// Created by carlostojal on 30-05-2023.
//

#include <tojalgrad/nn/layers/Layer.h>

namespace tojalgrad::nn::layers {

    Layer::Layer(int in_features, int out_features) {

        if(in_features <= 0)
            throw std::runtime_error("Invalid layer input size!");

        if(out_features <= 0)
            throw std::runtime_error("Invalid layer output size!");

        this->in_features = in_features;
        this->out_features = out_features;
    }

    int Layer::getInFeatures() const {
        return this->in_features;
    }

    int Layer::getOutFeatures() const {
        return this->out_features;
    }
} // layers