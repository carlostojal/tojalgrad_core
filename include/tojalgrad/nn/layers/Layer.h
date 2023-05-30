//
// Created by carlostojal on 30-05-2023.
//

#ifndef TOJALGRAD_CORE_LAYER_H
#define TOJALGRAD_CORE_LAYER_H

#include <eigen3/Eigen/Dense>

namespace tojalgrad::nn::layers {

    class Layer {
        protected:

            virtual Eigen::VectorXf forward(Eigen::VectorXf in) = 0;

            Layer *next = nullptr;
    };
} // layers

#endif //TOJALGRAD_CORE_LAYER_H
