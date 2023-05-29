//
// Created by carlostojal on 29-05-2023.
//

#include <gtest/gtest.h>
#include <tojalgrad/nn/Neuron.h>
#include <tojalgrad/nn/Activation.h>

using namespace tojalgrad::nn;

class NeuronActivationTests : public ::testing::Test {

    protected:
        Neuron neuron;

        Eigen::VectorXf input;

    public:
        NeuronActivationTests(): neuron(3, Activation::linear),
                                 input(3){
            input << 0.5f, 0.2f, 1.0f;
        }
};

// TODO: set seed on neuron random number generators to allow static testing

TEST_F(NeuronActivationTests, linearActivation) {

    ASSERT_EQ((int) (Activation::linear(12.3) * 100), 1230);
    ASSERT_EQ((int) (Activation::linear(51.87) * 100), 5187);
}

TEST_F(NeuronActivationTests, stepActivation) {

    ASSERT_EQ((int) Activation::step(0), 1);
    ASSERT_EQ((int) Activation::step(0.001), 1);
    ASSERT_EQ((int) Activation::step(-0.00001), 0);
    ASSERT_EQ((int) Activation::step(100002), 1);
    ASSERT_EQ((int) Activation::step(-111122), 0);
}

TEST_F(NeuronActivationTests, sigmoidActivation) {

    ASSERT_EQ((int) (Activation::sigmoid(0.72) * 100), 67);
    ASSERT_EQ((int) (Activation::sigmoid(15) * 100), 99);
    ASSERT_EQ((int) (Activation::sigmoid(-2) * 100), 11);
}

TEST_F(NeuronActivationTests, tanhActivation) {

    ASSERT_EQ((int) (Activation::tanh(1) * 100), 76);
    ASSERT_EQ((int) (Activation::tanh(-0.5) * 100), -46);
}

TEST_F(NeuronActivationTests, reluActivation) {

    ASSERT_EQ((int) Activation::ReLU(0), 0);
    ASSERT_EQ((int) Activation::ReLU(5), 5);
    ASSERT_EQ((int) Activation::ReLU(-0.00001), 0);
}

TEST_F(NeuronActivationTests, mismatchedSizeInput) {

    Eigen::VectorXf dummyInput = Eigen::VectorXf(4);
    dummyInput << 1, 2, 3, 4;

    ASSERT_THROW(neuron.forward(dummyInput), std::runtime_error);
}