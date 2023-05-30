//
// Created by carlostojal on 30-05-2023.
//

#ifndef TOJALGRAD_CORE_BOBNET_H
#define TOJALGRAD_CORE_BOBNET_H

#include "Model.h"

namespace tojalgrad::nn::models {

    /*! \brief BobNet neural network model.
     *
     * BobNet is a classic example of a Deep Neural Network (DNN) to classify the MNIST dataset.
     */
    class BobNet : Model {

        public:
            /*! \brief Class constructor. */
            BobNet();
    };

} // models

#endif //TOJALGRAD_CORE_BOBNET_H
