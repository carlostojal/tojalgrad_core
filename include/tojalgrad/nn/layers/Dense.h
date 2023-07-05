//
// Created by carlostojal on 30-05-2023.
//

#ifndef TOJALGRAD_CORE_DENSE_H
#define TOJALGRAD_CORE_DENSE_H

#include <functional>
#include <tojalgrad/nn/layers/Layer.h>
#include <tojalgrad/nn/Neuron.h>
#include <vector>
#include <thread>

namespace tojalgrad::nn::layers {

    /*! \brief Dense layer class. */
    class Dense : public Layer {

        private:
            /*! \brief Vector of neurons on this layer. */
            std::vector<tojalgrad::nn::Neuron> neurons;
            /*! \brief Number of inputs on each neuron. */
            int n_inputs = 0;
            /*! \brief Number of neurons on this layer. */
            int n_neurons = 0;
            /*! \brief Last input vector. */
            Eigen::VectorXf lastInput;

        public:
            /*! \brief Class constructor.
             *
             * @param in_features Number of input features (i.e. number of neurons on the previous layer.
             * @param out_features Number of output features (i.e. number of neurons on this layer.
             * @param activation Activation function. Examples at tojalgrad::nn::Activation.
             */
            Dense(int in_features, int out_features, const std::function<float(float)>& activation);

            /*! \brief Forward pass method.
             *
             * @param in Input vector.
             * @return Activation values vector.
             */
            Eigen::VectorXf forward(Eigen::VectorXf in) override;

            /*! \brief Compute the error of this layer. */
            void backPropagate(const Eigen::VectorXf& out, float learning_rate) override;

            Dense *next = nullptr;
            Dense *prev = nullptr;

    };
} // layers

#endif //TOJALGRAD_CORE_DENSE_H
