//
// Created by carlostojal on 29-05-2023.
//

#include <tojalgrad/nn/Neuron.h>

namespace tojalgrad::nn {

        Neuron::Neuron(int n_inputs, const std::function<float(float in)>& activation) {

            if(n_inputs <= 0)
                throw std::runtime_error("Invalid number of neuron inputs!");

            // allocate weights vector
            this->w = Eigen::VectorXf(n_inputs);

            // set activation function
            this->activation = activation;
        }

        float Neuron::forward(const Eigen::VectorXf& inputs) {

            if(this->activation == nullptr)
                throw std::runtime_error("No activation function was set!");

            if(inputs.size() != this->w.size())
                throw std::runtime_error("Mismatched input vector size!");

            // TODO: acceleration here
            float out = this->w.dot(inputs) + this->b;

            this->lastValue =  this->activation(out);

            return this->lastValue;
        }

    void Neuron::initRandomWeights() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> distrib(0.0f, 1.0f);

        int n_weights = this->w.size();
        if(n_weights == 0)
            throw std::runtime_error("Weights have no size");

        // lambda function to fill a given weight with a random value
        auto initWithRandomValue = [](int index, Eigen::VectorXf& weights,
                                      std::uniform_real_distribution<float>& distrib, std::mt19937& gen) {
            if (index < 0 || index >= weights.size())
                throw std::runtime_error("Initialization weight index out of bounds!");

            weights[index] = distrib(gen);
        };

        // create a thread to init each weight with a random value
        std::vector<std::thread> threadList;
        for (int i = 0; i < n_weights; i++) {
            threadList.emplace_back(initWithRandomValue, i, std::ref(this->w), std::ref(distrib), std::ref(gen));
        }

        // wait the threads
        for(auto & t : threadList) {
            t.join();
        }
    }

    float Neuron::getLastValue() {
        return this->lastValue;
    }

    Eigen::VectorXf Neuron::getWeights() const {
        return this->w;
    }

    void Neuron::setWeight(int index, float value) {

        if(index < 0 || index >= this->w.size())
            throw std::runtime_error("Tried to set an invalid weight index!");

        this->w[index] = value;
    }

    float Neuron::getBias() const {
        return this->b;
    }

    void Neuron::setBias(float bias) {
        this->b = bias;
    }

} // nn