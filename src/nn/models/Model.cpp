//
// Created by carlostojal on 30-05-2023.
//

#include <tojalgrad/nn/models/Model.h>

namespace tojalgrad::nn {

    Model::Model() {
        this->first = nullptr;
    }

    void Model::add(layers::Layer *layer) {

        // empty model
        if(this->first == nullptr) {
            this->first = layer;
            layer->prev = layer;
        } else { // add the layer to the tail
            layers::Layer *tail = this->first->prev;

            if(tail->getOutFeatures() != layer->getInFeatures())
                throw std::runtime_error("Mismatched layer feature dimensions!");

            this->first->prev = layer;
            tail->next = layer;
            layer->prev = tail;
            layer->next = nullptr;
        }
    }

    Eigen::VectorXf Model::forward(const Eigen::VectorXf& in) {

        if(in.size() != this->first->getInFeatures())
            throw std::runtime_error("Unexpected input size!");

        layers::Layer *iter = this->first;

        Eigen::VectorXf cur = in;

        while(iter != nullptr) {

            cur = iter->forward(cur);

            iter = iter->next;
        }

        return cur;
    }

    float Model::getLearningRate() const {
        return this->learning_rate;
    }

    void Model::setLearningRate(float learning_rate) {
        this->learning_rate = learning_rate;
    }
} // nn