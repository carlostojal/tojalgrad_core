//
// Created by carlostojal on 29-05-2023.
//

#ifndef TOJALGRAD_CORE_ACTIVATION_H
#define TOJALGRAD_CORE_ACTIVATION_H

#include <math.h>

namespace tojalgrad::nn {

    /*! \brief Collection of activation functions. */
    class Activation {

        public:
            /*! \brief Linear activation. */
            static float linear(float in);

            /*! \brief Step activation. */
            static float step(float in);

            /*! \brief Signal activation. */
            static float sign(float in);

            /*! \brief Sigmoid activation. */
            static float sigmoid(float in);

            /*! \brief Tanh activation. */
            static float tanh(float in);

            /*! \brief ReLU activation. */
            static float ReLU(float in);
    };

} // nn

#endif //TOJALGRAD_CORE_ACTIVATION_H
