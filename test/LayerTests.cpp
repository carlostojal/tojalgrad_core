//
// Created by carlostojal on 30-05-2023.
//

#include <gtest/gtest.h>
#include <tojalgrad/nn/layers/Layer.h>
#include <tojalgrad/nn/layers/Dense.h>
#include <tojalgrad/nn/Activation.h>

using namespace tojalgrad::nn;

class LayerTests : public ::testing::Test {


};

TEST_F(LayerTests, linearLayerInstantiation) {

    EXPECT_NO_THROW(layers::Dense linear = layers::Dense(2, 5, Activation::sigmoid));
}

TEST_F(LayerTests, invalidLayerInstantiation) {

    EXPECT_THROW(layers::Dense linear = layers::Dense(-1, -5, Activation::sigmoid), std::runtime_error);
}

TEST_F(LayerTests, invalidInputLayerForward) {

    // this layer expects a vector of size 2 as input
    layers::Dense linear = layers::Dense(2, 5, Activation::sigmoid);

    Eigen::VectorXf vec(3); // size 3: invalid
    vec << 1.2f, 1.4f, 2.4f;

    EXPECT_THROW(linear.forward(vec), std::runtime_error);
}

TEST_F(LayerTests, validInputLayerForward) {

    // this layer expects a vector of size 2 as input
    layers::Dense linear = layers::Dense(2, 5, Activation::sigmoid);

    Eigen::VectorXf vec(2); // size 2
    vec << 1.2f, 1.4f;

    EXPECT_NO_THROW(linear.forward(vec));
}