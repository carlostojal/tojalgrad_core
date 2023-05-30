//
// Created by carlostojal on 30-05-2023.
//

#include <gtest/gtest.h>
#include <tojalgrad/nn/Loss.h>

using namespace tojalgrad::nn;

class LossTests : public ::testing::Test {


};

TEST_F(LossTests, mseLoss) {

    Eigen::Vector3f v1(2, 5, 1);
    Eigen::Vector3f v2(1, 3, 4);

    float loss = Loss::MSE(v1, v2);

    ASSERT_EQ((int) (loss * 100), 466);
}

TEST_F(LossTests, crossEntropyLoss) {

    Eigen::Vector4f v1(0, 1.0f, 0, 0);
    Eigen::Vector4f v2(0.5f, 0.7f, 0.1f, 0.2f);

    float loss = Loss::CrossEntropy(v1, v2);

    ASSERT_EQ((int) (loss * 100), 51);
}
