//
// Created by carlostojal on 29-05-2023.
//

#ifndef TOJALGRAD_CORE_NEURON_H
#define TOJALGRAD_CORE_NEURON_H

#include <random>
#include <functional>
#include <vector>
#include <thread>
#include <eigen3/Eigen/Dense>

namespace tojalgrad::nn {

        /*!
         * \brief Basic artificial neuron class.
         *
         * Receives an activation function, keeps weights, bias and the last activation value.
         */
        class Neuron {

            private:
                // weights and bias
                /*! \brief Weights vector. One weight for each connection on the previous layer. Starts randomly. */
                Eigen::VectorXf w;
                /*! \brief Bias value. Starts at zero. */
                float b = 0;

                /*! \brief The last activation value. */
                float lastValue = 0;

                /*! \brief Neuron activation function. Some examples on tojalgrad::nn::Activation. */
                std::function<float(float in)> activation = nullptr;

                /*! \brief Method to initialize the neuron with random weight values (between 0 and 1). */
                void initRandomWeights();

            public:
                /*! \brief Construct a Neuron.
                 *
                 * @param n_inputs Number of connections to the previous layer.
                 * @param activation Activation function. Examples on tojalgrad::nn::Activation.
                 */
                explicit Neuron(int n_inputs, const std::function<float(float in)>& activation);

                /*! \brief Activate the neuron.
                 *
                 * @param inputs The vector of inputs to the neuron.
                 * @return The activation value after passing to the activation function.
                 */
                float forward(const Eigen::VectorXf& inputs);

                /*! \brief Get the last activation value.
                 *
                 * @return The last activation value.
                 */
                float getLastValue();
        };
        
} // nn

#endif //TOJALGRAD_CORE_NEURON_H
