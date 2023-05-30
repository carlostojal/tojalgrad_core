//
// Created by carlostojal on 30-05-2023.
//

#ifndef TOJALGRAD_CORE_LAYER_H
#define TOJALGRAD_CORE_LAYER_H

#include <eigen3/Eigen/Dense>

namespace tojalgrad::nn::layers {

    /*! \brief Abstract layer class. Represents a neural network layer. */
    class Layer {
        protected:
            /*! \brief Class constructor.
             *
             * @param in_features Number of input features (for example, number of neurons on the previous layer).
             * @param out_features Number of output features (i.e. the number of neurons of this layer).
             */
            Layer(int in_features, int out_features);

            /*! \brief Number of input neurons. */
            int in_features;
            /*! \brief Number of generated outputs (neurons in this layer). */
            int out_features;

        public:
            /*! \brief Pointer to the next layer, if inserted in a network. */
            Layer *next = nullptr;
            /*! \brief Pointer to the previous layer, if inserted in a network. */
            Layer *prev = nullptr;

            /*! \brief Layer forward pass.
             *
             * @param in Input vector.
             * @return Vector of neuron activations of this layer.
             */
            virtual Eigen::VectorXf forward(Eigen::VectorXf in) = 0;

            /*! \brief Get number of input features. */
            int getInFeatures() const;
            /*! \brief Get number of output features. */
            int getOutFeatures() const;
    };
} // layers

#endif //TOJALGRAD_CORE_LAYER_H
