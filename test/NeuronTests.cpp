//
// Created by carlostojal on 29-05-2023.
//

#include <gtest/gtest.h>
#include <tojalgrad/nn/Neuron.h>

class NeuronTests : public ::testing::Test {

    protected:
        tojalgrad::nn::Neuron n1;

        void SetUp() override {

        }
};

TEST_F(NeuronTests, forwardFeed) {

    Eigen::VectorXf input(3);
    input << 0.5f, 0.2f, 1.0f;

    // TODO: write activation function
    FAIL();
}