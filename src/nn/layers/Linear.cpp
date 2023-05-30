//
// Created by carlostojal on 30-05-2023.
//

#include <tojalgrad/nn/layers/Linear.h>

namespace tojalgrad::nn::layers {

    Linear::Linear(int in_features, int out_features, const std::function<float(float)>& activation) :
        Layer(in_features, out_features) {

        // init the neurons
        for(int i = 0; i < out_features; i++)
            this->neurons.emplace_back(in_features, activation);
    }

    Eigen::VectorXf Linear::forward(Eigen::VectorXf in) {

        // check input size with the number of neuron inputs
        if(in.size() != this->in_features)
            throw std::runtime_error("Unexpected input size!");

        Eigen::VectorXf out(this->out_features);

        // activate each neuron of the layer with the input
        // TODO: acceleration here
        for(int i = 0; i < this->out_features; i++)
            out[i] = this->neurons[i].forward(in);

        return out;
    }

} // layers