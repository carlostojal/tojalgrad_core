//
// Created by carlostojal on 30-05-2023.
//

#include <gtest/gtest.h>
#include <tojalgrad/nn/Neuron.h>
#include <tojalgrad/nn/Activation.h>

using namespace tojalgrad::nn;

class NeuronTrainingTests : public ::testing::Test {

};

TEST_F(NeuronTrainingTests, andGate) {

    float learning_rate = 0.5;

    Neuron neuron(2, Activation::step);

    std::cout.precision(5);

    std::cout << "Initial weights: " << neuron.getWeights() << "\n";

    for(int epoch = 0; epoch < 100; epoch++) {

        std::cout << "Epoch " << epoch + 1 << "/" << 10 << ":\n";

        // generate ground truth
        for (int sample = 0b00; sample <= 0b11; sample++) {

            // bit mask
            int x1 = (sample & 0b10) >> 1;
            int x2 = sample & 0b01;

            // compute the expected output (and gate)
            int expected = x1 & x2;

            float output = neuron.forward(Eigen::Vector2f(x1, x2));

            float error = (float) expected - output;

            std::cout << "Inputs: " << std::fixed << x1 << " " << std::fixed << x2 << " | Expected: " << std::fixed <<
            expected << " | Output: " << std::fixed << output << " | Error: " << std::fixed << error << std::endl;

            // set the neuron error and update its weights and bias
            neuron.setError(error);
            neuron.setWeight(0, neuron.getWeights()[0] + (learning_rate * error * (float) x1));
            neuron.setWeight(1, neuron.getWeights()[1] + (learning_rate * error * (float) x2));
            neuron.setBias(1);

            // std::cout << neuron.getWeights() << std::endl;
        }
    }

    std::cout << "Final weights: " << neuron.getWeights() << "\n";

    for(int i = 0; i <= 0b11; i++) {
        ASSERT_EQ((int) neuron.forward(Eigen::Vector2f((i & 0b10) >> 1, i & 0b01)), i);
    }
}