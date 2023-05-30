//
// Created by carlostojal on 30-05-2023.
//

#include <tojalgrad/nn/models/Model.h>
#include <tojalgrad/nn/models/BobNet.h>
#include <tojalgrad/nn/layers/Linear.h>
#include <tojalgrad/nn/Activation.h>

namespace tojalgrad::nn::models {

    BobNet::BobNet() : Model() {

        layers::Linear l1 = layers::Linear(784, 800, Activation::sigmoid);
        layers::Linear l2 = layers::Linear(800, 10, Activation::sigmoid);

        this->add(&l1);
        this->add(&l2);
    }
} // models