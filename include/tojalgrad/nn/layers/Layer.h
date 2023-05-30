//
// Created by carlostojal on 30-05-2023.
//

#ifndef TOJALGRAD_CORE_LAYER_H
#define TOJALGRAD_CORE_LAYER_H

#include <eigen3/Eigen/Dense>

namespace tojalgrad::nn::layers {

    class Layer {
        protected:
            Layer(int in_features, int out_features);

            int in_features;
            int out_features;

        public:
            Layer *next = nullptr;
            Layer *prev = nullptr;

            virtual Eigen::VectorXf forward(Eigen::VectorXf in) = 0;

            int getInFeatures() const;
            int getOutFeatures() const;
    };
} // layers

#endif //TOJALGRAD_CORE_LAYER_H
