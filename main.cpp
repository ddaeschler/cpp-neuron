#include "connection.h"
#include "neuron.h"
#include "layer_tools.h"

#include <iostream>
#include <cmath>
#include <vector>

#include <Eigen/Dense>

double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double relu(double x) {
    return std::max(0.0, x);
}

void print(double f) {
    std::cout << f << std::endl;
}

static std::vector<Connection<double>> cp;
Connection<double>& getFromPool() {
    if (cp.size() == 0) cp.reserve(30000); //make sure a realloc doesnt happen

    cp.push_back(Connection<double>(1.0));
    return cp.back();
}



int main() {
    // Single neuron with a single input
    auto l1 = BuildLayer<double>(1, 2, 1, relu);
    auto l2 = BuildLayer<double>(2, 100, 2, relu);
    auto l3 = BuildLayer<double>(3, 100, 100, relu);
    auto l4 = BuildLayer<double>(4, 2, 100, relu);

    FullyConnectLayers<std::vector<Neuron<double>>, std::function<Connection<double>&()>, double>(l1, l2, &getFromPool);
    FullyConnectLayers<std::vector<Neuron<double>>, std::function<Connection<double>&()>, double>(l2, l3, &getFromPool);
    FullyConnectLayers<std::vector<Neuron<double>>, std::function<Connection<double>&()>, double>(l3, l4, &getFromPool);

    l4[0].addOutput(print);
    l4[1].addOutput(print);

    l1[0].signal(1);
    l1[1].signal(4);

    return 0;
}
