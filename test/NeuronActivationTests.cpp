//
// Created by carlostojal on 29-05-2023.
//

#include <gtest/gtest.h>
#include <tojalgrad/nn/Neuron.h>
#include <tojalgrad/nn/Activation.h>

using namespace tojalgrad::nn;

class NeuronActivationTests : public ::testing::Test {

    protected:
        Neuron linearNeuron;
        Neuron stepNeuron;
        Neuron sigmoidNeuron;
        Neuron reluNeuron;

        Eigen::VectorXf input;

    public:
        NeuronActivationTests(): linearNeuron(3, Activation::linear),
                                 stepNeuron(3, Activation::step),
                                 sigmoidNeuron(3, Activation::sigmoid),
                                 reluNeuron(3, Activation::ReLU),
                                 input(3){
            input << 0.5f, 0.2f, 1.0f;
        }
};

// TODO: set seed on neuron random number generators to allow static testing

TEST_F(NeuronActivationTests, linearForwardFeed) {

    float out = linearNeuron.forward(input);

    // TODO: compare output with expected
    FAIL();
}

TEST_F(NeuronActivationTests, stepForwardFeed) {

    float out = stepNeuron.forward(input);

    FAIL();
}

TEST_F(NeuronActivationTests, sigmoidForwardFeed) {

    float out = sigmoidNeuron.forward(input);

    FAIL();
}

TEST_F(NeuronActivationTests, reluForwardFeed) {

    float out = reluNeuron.forward(input);

    FAIL();
}

TEST_F(NeuronActivationTests, mismatchedSizeInput) {

    Eigen::VectorXf dummyInput = Eigen::VectorXf(4);
    dummyInput << 1, 2, 3, 4;

    ASSERT_THROW(linearNeuron.forward(dummyInput), std::runtime_error);
}