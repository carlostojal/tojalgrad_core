//
// Created by carlostojal on 29-05-2023.
//

#ifndef TOJALGRAD_CORE_NEURON_H
#define TOJALGRAD_CORE_NEURON_H

#include <random>
#include <functional>
#include <vector>
#include <thread>
#include <eigen3/Eigen/Dense>

namespace tojalgrad {
    namespace nn {

        class Neuron {

            private:
                // weights and bias
                Eigen::VectorXf w;
                float b = 0;

                float lastValue = 0;

                std::function<float(float in)> activation = nullptr;

                void initRandomWeights();

            public:
                explicit Neuron(int n_inputs, const std::function<float(float in)>& activation);

                float forward(const Eigen::VectorXf& inputs);

                float getLastValue();
        };

    } // tojalgrad
} // nn

#endif //TOJALGRAD_CORE_NEURON_H
