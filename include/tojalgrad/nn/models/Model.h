//
// Created by carlostojal on 30-05-2023.
//

#ifndef TOJALGRAD_CORE_MODEL_H
#define TOJALGRAD_CORE_MODEL_H

#include <vector>
#include <tojalgrad/nn/layers/Layer.h>

namespace tojalgrad::nn {

    /*! \brief Neural network model class. Represents any neural network model. */
    class Model {

        private:
            /*! \brief Pointer to the first layer of the network. */
            tojalgrad::nn::layers::Layer *first = nullptr;

        public:
            /*! \brief Class constructor */
            Model();

            /*! \brief Add a layer to the network.
             *
             * @param layer Pointer to the layer to add.
             */
            void add(tojalgrad::nn::layers::Layer *layer);

            /*! \brief Forward pass method. Iterates every layer, activating it and propagating.
             *
             * @param in Input vector
             * @return Activation vector.
             */
            Eigen::VectorXf forward(const Eigen::VectorXf& in);
    };

} // nn

#endif //TOJALGRAD_CORE_MODEL_H
