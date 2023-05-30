//
// Created by carlostojal on 30-05-2023.
//

#ifndef TOJALGRAD_CORE_LOSS_H
#define TOJALGRAD_CORE_LOSS_H

#include <eigen3/Eigen/Dense>
#include <thread>
#include <mutex>

namespace tojalgrad {
    namespace nn {

        /*! \brief Collection of loss functions. */
        class Loss {

            public:
                static float MSE(const Eigen::VectorXf &ground_truth, const Eigen::VectorXf &prediction);
                static float CrossEntropy(const Eigen::VectorXf& ground_truth, const Eigen::VectorXf& prediction);

        };

    } // tojalgrad
} // nn

#endif //TOJALGRAD_CORE_LOSS_H
