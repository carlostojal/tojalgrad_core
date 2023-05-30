//
// Created by carlostojal on 30-05-2023.
//

#include <tojalgrad/nn/Loss.h>

namespace tojalgrad::nn {

    float Loss::MSE(const Eigen::VectorXf &v1, const Eigen::VectorXf &v2) {

        if(v1.size() != v2.size())
            throw std::runtime_error("Can't compute error: mismatched sizes!");

        float error = 0;

        

        // TODO: acceleration here
        for(int i = 0; i < v1.size(); i++) {
            // compute squared error
            float e1 = pow(v1[i] - v2[i], 2);
            // contribute to the mse error
            error += e1 * ((float) 1 / v1.size());
        }

        return error;
    }
} // nn