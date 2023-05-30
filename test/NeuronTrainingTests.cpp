//
// Created by carlostojal on 30-05-2023.
//

#include <gtest/gtest.h>
#include <tojalgrad/nn/Neuron.h>
#include <tojalgrad/nn/Activation.h>

using namespace tojalgrad::nn;

class NeuronTrainingTests : public ::testing::Test {

};

TEST_F(NeuronTrainingTests, perceptronANDGate) {

    float learning_rate = 1;

    Neuron neuron(2, Activation::step);

    std::cout.precision(5);

    int epoch = 0;

    int right_samples;

    std::cout << "Initial weights: " << neuron.getWeights() << std::endl;

    do {

        right_samples = 0;
        std::cout << "\nEpoch " << epoch+1 << std::endl;

        // generate ground truth
        for (int sample = 0b00; sample <= 0b11; sample++) {

            // bit mask
            int x1 = (sample & 0b10) >> 1;
            int x2 = sample & 0b01;

            // compute the expected output (and gate)
            int expected = x1 & x2;

            // compute the output
            float output = neuron.forward(Eigen::Vector2f(x1, x2));

            // compute the error
            int error = expected - (int) output;

            std::cout << "Inputs: " << std::fixed << x1 << " " << std::fixed << x2 << " | Expected: " << std::fixed <<
            expected << " | Output: " << std::fixed << output << " | Error: " << std::fixed << error << std::endl;

            // if the error is 0, the sample is correct, don't adjust the weights
            if(error == 0) {
                right_samples++;
                continue;
            }

            // set the neuron error and update its weights and bias
            neuron.setError(error);
            neuron.setWeight(0, neuron.getWeights()[0] + (learning_rate * error * (float) x1));
            neuron.setWeight(1, neuron.getWeights()[1] + (learning_rate * error * (float) x2));
            neuron.setBias(neuron.getBias() + (learning_rate * error));

            // std::cout << neuron.getWeights() << std::endl;
        }

        epoch++;

        // std::cout << "Right samples: " << right_samples << "\n";

    } while(right_samples != 4);

    std::cout << "\nFinal weights: " << neuron.getWeights() << " Bias: " << neuron.getBias() << std::endl;

    for(int i = 0; i <= 0b11; i++) {
        int x1 = (i & 0b10) >> 1;
        int x2 = i & 0b01;

        ASSERT_EQ((int) neuron.forward(Eigen::Vector2f(x1, x2)), x1 & x2);
    }
}