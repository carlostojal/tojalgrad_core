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
