//
// Created by carlostojal on 30-05-2023.
//

#include <tojalgrad/nn/Model.h>

namespace tojalgrad::nn {

    void Model::add(layers::Layer *layer) {

        // empty model
        if(this->first == nullptr) {
            this->first = layer;
            layer->prev = layer;
        } else { // add the layer to the tail
            layers::Layer *tail = this->first->prev;
            tail->next = layer;
            layer->prev = tail;
            layer->next = this->first;
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
} // nn