//
// Created by David Daeschler on 4/7/23.
//

#ifndef CPP_NEURON_LAYER_TOOLS_H
#define CPP_NEURON_LAYER_TOOLS_H

#include "connection.h"
#include "neuron.h"

#include <functional>
#include <sstream>

template<typename LT, typename Pool, typename NT>
void FullyConnectLayers(LT& layer1, LT& layer2, Pool p) {
    for (Neuron<NT>& l1obj : layer1) {
        for (Neuron<NT>& l2obj : layer2) {
            Connection<NT>& conn = p();
            l1obj.addOutput([&conn](NT value) {
                return conn.signal(value);
            });
            conn.addOutput([&l2obj](NT value) {
                return l2obj.signal(value);
            });
        }
    }
}

template<typename NT>
std::vector<Neuron<NT>> BuildLayer(int layerNumber, int nodeCount, int numInputs,
                                   std::function<double(double)> activationFunction) {
    std::vector<Neuron<NT>> layer;

    for (int i = 0; i < nodeCount; ++i) {
        std::stringstream name;
        name << "l" << layerNumber << "n" << i;

        layer.push_back(Neuron<double>(activationFunction, numInputs, name.str()));
    }

    return layer;
}

#endif //CPP_NEURON_LAYER_TOOLS_H
