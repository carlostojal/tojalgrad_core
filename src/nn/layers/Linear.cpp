//
// Created by carlostojal on 30-05-2023.
//

#include <tojalgrad/nn/layers/Linear.h>

namespace tojalgrad::nn::layers {

    Linear::Linear(int input_neurons, int output_neurons, const std::function<float(float)>& activation) {

        this->n_inputs = input_neurons;
        this->n_neurons = output_neurons;

        // init the neurons
        for(int i = 0; i < output_neurons; i++)
            this->neurons.emplace_back(input_neurons, activation);
    }

    Eigen::VectorXf Linear::forward(Eigen::VectorXf in) {

        // check input size with the number of neuron inputs
        if(in.size() != this->n_inputs)
            throw std::runtime_error("Unexpected input size!");

        Eigen::VectorXf out(this->n_neurons);

        // activate each neuron of the layer with the input
        for(int i = 0; i < this->n_neurons; i++)
            out[i] = this->neurons[i].forward(in);

        return out;
    }

} // layers