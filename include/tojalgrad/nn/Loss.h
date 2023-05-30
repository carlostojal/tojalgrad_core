//
// Created by carlostojal on 30-05-2023.
//

#ifndef TOJALGRAD_CORE_LOSS_H
#define TOJALGRAD_CORE_LOSS_H

#include <eigen3/Eigen/Dense>

namespace tojalgrad {
    namespace nn {

        /*! \brief Collection of loss functions. */
        class Loss {

            public:
                static float MSE(const Eigen::VectorXf &v1, const Eigen::VectorXf &v2);

        };

    } // tojalgrad
} // nn

#endif //TOJALGRAD_CORE_LOSS_H
