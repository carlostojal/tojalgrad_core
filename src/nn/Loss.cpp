//
// Created by carlostojal on 30-05-2023.
//

#include <tojalgrad/nn/Loss.h>

namespace tojalgrad::nn {

    float Loss::MSE(const Eigen::VectorXf &v1, const Eigen::VectorXf &v2) {

        if(v1.size() != v2.size())
            throw std::runtime_error("Can't compute error: mismatched sizes!");

        float error = 0;

        std::mutex errorMutex;

        auto mseLossRoutine = [](int index, const Eigen::VectorXf& v1, const Eigen::VectorXf& v2, float *error,
                std::mutex *errorMutex) {

            // compute squared error
            float e1 = pow(v1[index] - v2[index], 2);
            // contribute to the mse error
            e1 *= ((float) 1 / v1.size());

            // lock access to the error
            errorMutex->lock();
            *error += e1;
            errorMutex->unlock();
        };

        std::vector<std::thread> threads;

        // TODO: acceleration here
        for(int i = 0; i < v1.size(); i++)
            threads.emplace_back(mseLossRoutine, i, v1, v2, &error, &errorMutex);

        // wait for all the threads
        for(auto & t : threads)
            t.join();

        return error;
    }
} // nn