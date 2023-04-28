//
// Created by David Daeschler on 4/7/23.
//

#ifndef CPP_NEURON_COMMON_H
#define CPP_NEURON_COMMON_H

#include <concepts>

typedef float fp_t;

template<class T> concept Integral = std::is_integral<T>::value;

#endif //CPP_NEURON_COMMON_H
