//
// Created by David Daeschler on 4/7/23.
//

#ifndef CPP_NEURON_COMMON_H
#define CPP_NEURON_COMMON_H

#include <concepts>

template<typename T>
concept floating_point = std::floating_point<T>;

#endif //CPP_NEURON_COMMON_H
