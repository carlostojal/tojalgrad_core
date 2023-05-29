//
// Created by carlostojal on 29-05-2023.
//

#ifndef TOJALGRAD_CORE_ACTIVATION_H
#define TOJALGRAD_CORE_ACTIVATION_H

#include <math.h>

namespace tojalgrad::nn {

        class Activation {

            public:
                static float linear(float in);
                static float step(float in);
                static float sigmoid(float in);
                static float ReLU(float in);
        };

} // nn

#endif //TOJALGRAD_CORE_ACTIVATION_H
