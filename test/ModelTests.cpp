//
// Created by carlostojal on 30-05-2023.
//

#include <gtest/gtest.h>
#include <tojalgrad/nn/models/Model.h>
#include <tojalgrad/nn/models/BobNet.h>
#include <tojalgrad/nn/layers/Dense.h>
#include <tojalgrad/nn/Activation.h>

using namespace tojalgrad::nn;

class ModelTests : public ::testing::Test {


};

TEST_F(ModelTests, createValidModel) {

    models::Model m = models::Model();

    layers::Dense l1 = layers::Dense(2, 3, Activation::ReLU);
    layers::Dense l2 = layers::Dense(3, 5, Activation::ReLU);
    layers::Dense l3 = layers::Dense(5, 8, Activation::ReLU);
    layers::Dense l4 = layers::Dense(8, 1, Activation::ReLU);

    ASSERT_NO_THROW(m.add(&l1));
    ASSERT_NO_THROW(m.add(&l2));
    ASSERT_NO_THROW(m.add(&l3));
    ASSERT_NO_THROW(m.add(&l4));
}

TEST_F(ModelTests, createInvalidModel) {

    models::Model m = models::Model();

    layers::Dense l1 = layers::Dense(2, 3, Activation::ReLU);
    layers::Dense l2 = layers::Dense(3, 5, Activation::ReLU);
    layers::Dense l3 = layers::Dense(7, 8, Activation::ReLU); // input should be 5

    ASSERT_NO_THROW(m.add(&l1));
    ASSERT_NO_THROW(m.add(&l2));
    ASSERT_THROW(m.add(&l3), std::runtime_error);
}

TEST_F(ModelTests, validForwardPass) {
    models::Model m = models::Model();

    layers::Dense l1 = layers::Dense(2, 3, Activation::ReLU);
    layers::Dense l2 = layers::Dense(3, 5, Activation::ReLU);
    layers::Dense l3 = layers::Dense(5, 8, Activation::ReLU);
    layers::Dense l4 = layers::Dense(8, 1, Activation::ReLU);

    m.add(&l1);
    m.add(&l2);
    m.add(&l3);
    m.add(&l4);

    Eigen::VectorXf in(2);
    in << 1.2f, 5.22f;

    Eigen::VectorXf out;
    ASSERT_NO_THROW(out = m.forward(in));

    ASSERT_EQ(out.size(), 1);
}

TEST_F(ModelTests, invalidForwardPass) {

    models::Model m = models::Model();

    layers::Dense l1 = layers::Dense(2, 3, Activation::ReLU);
    layers::Dense l2 = layers::Dense(3, 5, Activation::ReLU);
    layers::Dense l3 = layers::Dense(5, 8, Activation::ReLU);
    layers::Dense l4 = layers::Dense(8, 1, Activation::ReLU);

    m.add(&l1);
    m.add(&l2);
    m.add(&l3);
    m.add(&l4);

    Eigen::VectorXf in(1);
    in << 1.2f;
    ASSERT_THROW(m.forward(in), std::runtime_error);
}

TEST_F(ModelTests, bobNetInstantiation) {

    ASSERT_NO_THROW(models::BobNet bobNet = models::BobNet());
}