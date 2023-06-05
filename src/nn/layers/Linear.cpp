//
// Created by carlostojal on 30-05-2023.
//

#include <tojalgrad/nn/layers/Dense.h>

namespace tojalgrad::nn::layers {

    Dense::Dense(int in_features, int out_features, const std::function<float(float)>& activation) :
        Layer(in_features, out_features) {

        // pre-allocate the neurons vector. enables parallelization without mutexes below
        this->neurons.resize(out_features);

        auto neuronInit = [](int index, int in_features, const std::function<float(float)>& activation,
                std::vector<Neuron>& neurons) {

            neurons[index] = Neuron(in_features, activation);
        };

        std::vector<std::thread> threads;

        // TODO: acceleration here
        // init the neurons
        for(int i = 0; i < out_features; i++) {
            // calling all these threads is leading the system out of resources
            // threads.emplace_back(neuronInit, i, in_features, std::ref(activation), std::ref(this->neurons));
            neuronInit(i, in_features, activation, this->neurons);
        }

        // wait the threads
        /*
        for(auto & t : threads)
            t.join();
            */
    }

    Eigen::VectorXf Dense::forward(Eigen::VectorXf in) {

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

    Eigen::VectorXf Dense::backPropagate() {

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