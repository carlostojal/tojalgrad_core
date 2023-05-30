//
// Created by carlostojal on 30-05-2023.
//

#include <tojalgrad/nn/Loss.h>

namespace tojalgrad::nn {

    float Loss::MSE(const Eigen::VectorXf &ground_truth, const Eigen::VectorXf &prediction) {

        if(ground_truth.size() != prediction.size())
            throw std::runtime_error("Can't compute error: mismatched sizes!");

        float error = 0;

        std::mutex errorMutex;

        auto mseLossRoutine = [](int index, const Eigen::VectorXf& ground_truth, const Eigen::VectorXf& prediction, float *error,
                std::mutex *errorMutex) {

            // compute squared error
            float e1 = pow(ground_truth[index] - prediction[index], 2);
            // contribute to the mse error
            e1 *= ((float) 1 / prediction.size());

            // lock access to the error
            errorMutex->lock();
            *error += e1;
            errorMutex->unlock();
        };

        std::vector<std::thread> threads;

        // TODO: acceleration here
        for(int i = 0; i < prediction.size(); i++)
            threads.emplace_back(mseLossRoutine, i, ground_truth, prediction, &error, &errorMutex);

        // wait for all the threads
        for(auto & t : threads)
            t.join();

        return error;
    }

    float Loss::CrossEntropy(const Eigen::VectorXf &ground_truth, const Eigen::VectorXf &prediction) {

        if(ground_truth.size() != prediction.size())
            throw std::runtime_error("Can't compute loss: mismatched sizes!");

        float error = 0;

        std::mutex errorMutex;

        auto crossEntropyLossRoutine = [](int index, const Eigen::VectorXf& ground_truth, const Eigen::VectorXf& prediction,
                float *error, std::mutex *errorMutex) {

            float e1 = ground_truth[index] * log2f(prediction[index]);

            errorMutex->lock();
            *error += e1;
            errorMutex->unlock();
        };

        std::vector<std::thread> threads;

        // TODO: acceleration here
        for(int i = 0; i < ground_truth.size(); i++) {
            threads.emplace_back(crossEntropyLossRoutine, i, std::ref(ground_truth), std::ref(prediction), &error,
                                 &errorMutex);
        }

        for(auto & t : threads)
            t.join();

        return -error;
    }
} // nn