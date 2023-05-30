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
        std::vector<std::thread> threads;

        auto neuronForward = [](int index, const Eigen::VectorXf& in, Eigen::VectorXf& out,
                std::vector<Neuron>& neurons) {
            out[index] = neurons[index].forward(in);
        };

        // activate each neuron of the layer with the input
        // TODO: acceleration here
        for(int i = 0; i < this->out_features; i++)
            threads.emplace_back(neuronForward, i, std::ref(in), std::ref(out), std::ref(this->neurons));

        // wait for the threads
        for(auto & t : threads) {
            t.join();
        }

        return out;
    }

    Eigen::VectorXf Linear::backPropagate() {

        // TODO: this is the last layer, use the loss
        if(this->next == nullptr) {
            throw std::runtime_error("Can't backpropagate as this is the last layer!");
        }

        Eigen::VectorXf errors(this->n_neurons);

        // compute weighted sum error for each neuron
        for(auto & n : this->neurons) {

            // TODO

        }

        return errors;
    }

} // layers