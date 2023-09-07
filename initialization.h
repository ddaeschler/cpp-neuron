//
// Created by David Daeschler on 4/20/23.
//

#ifndef CPP_NEURON_INITIALIZATION_H
#define CPP_NEURON_INITIALIZATION_H

#include "common.h"
#include <random>
#include <Eigen/Dense>


class Initialization {
public:
    template<typename Derived>
    static void WeightInitHE(Eigen::MatrixBase<Derived>& matrix) {
        static_assert(std::is_same<typename Derived::Scalar, fp_t>::value, "Matrix must have fp_t scalar type");

        // He initialization standard deviation
        fp_t stddev = sqrt(2.0 / static_cast<fp_t>(matrix.cols()));

        // Create normal distribution
        std::normal_distribution<fp_t> dist(0.0, stddev);

        // Random number generator
        std::random_device rd;
        std::mt19937 gen(rd());

        // Assign random numbers to the matrix elements
        matrix = Derived::NullaryExpr(matrix.rows(), matrix.cols(), [&]() { return dist(gen); });
    }

    template<typename Derived>
    static void WeightInitRandom(Eigen::MatrixBase<Derived>& matrix) {
        static_assert(std::is_same<typename Derived::Scalar, fp_t>::value, "Matrix must have fp_t scalar type");
        matrix.setRandom();
    }
};


#endif //CPP_NEURON_INITIALIZATION_H
