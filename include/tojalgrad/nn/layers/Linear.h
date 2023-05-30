//
// Created by carlostojal on 30-05-2023.
//

#ifndef TOJALGRAD_CORE_LINEAR_H
#define TOJALGRAD_CORE_LINEAR_H

#include <functional>
#include <tojalgrad/nn/layers/Layer.h>
#include <tojalgrad/nn/Neuron.h>
#include <vector>
#include <thread>

namespace tojalgrad::nn::layers {

    class Linear : public Layer {

        private:
            std::vector<tojalgrad::nn::Neuron> neurons;
            int n_inputs = 0;
            int n_neurons = 0;

        public:
            Linear(int in_features, int out_features, const std::function<float(float)>& activation);

            Eigen::VectorXf forward(Eigen::VectorXf in) override;

    };
} // layers

#endif //TOJALGRAD_CORE_LINEAR_H
