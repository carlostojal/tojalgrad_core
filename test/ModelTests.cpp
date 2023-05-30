//
// Created by carlostojal on 30-05-2023.
//

#include <gtest/gtest.h>
#include <tojalgrad/nn/Model.h>
#include <tojalgrad/nn/layers/Linear.h>
#include <tojalgrad/nn/Activation.h>

using namespace tojalgrad::nn;

class ModelTests : public ::testing::Test {


};

TEST_F(ModelTests, createValidModel) {

    Model m = Model();

    layers::Linear l1 = layers::Linear(2, 3, Activation::ReLU);
    layers::Linear l2 = layers::Linear(3, 5, Activation::ReLU);
    layers::Linear l3 = layers::Linear(5, 8, Activation::ReLU);
    layers::Linear l4 = layers::Linear(8, 1, Activation::ReLU);

    EXPECT_NO_THROW(m.add(&l1));
    EXPECT_NO_THROW(m.add(&l2));
    EXPECT_NO_THROW(m.add(&l3));
    EXPECT_NO_THROW(m.add(&l4));
}