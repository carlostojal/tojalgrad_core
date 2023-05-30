//
// Created by carlostojal on 30-05-2023.
//

#ifndef TOJALGRAD_CORE_MODEL_H
#define TOJALGRAD_CORE_MODEL_H

#include <vector>
#include <tojalgrad/nn/layers/Layer.h>

namespace tojalgrad::nn {

    class Model {

        private:
            tojalgrad::nn::layers::Layer *first = nullptr;

        public:
            Model();

            void add(tojalgrad::nn::layers::Layer *layer);

            Eigen::VectorXf forward(const Eigen::VectorXf& in);
    };

} // nn

#endif //TOJALGRAD_CORE_MODEL_H
