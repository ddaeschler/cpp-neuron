//
// Created by David Daeschler on 4/19/23.
//

#ifndef CPP_NEURON_ACTIVATION_H
#define CPP_NEURON_ACTIVATION_H

#include "common.h"
#include <utility>
#include <Eigen/Dense>

template <typename T>
struct activation_traits;

template <typename T>
using matrix_activation_t = typename activation_traits<T>::matrix_type;

template <typename T>
using fp_activation_t = typename activation_traits<T>::fp_type;

template<typename T>
concept HasActivation = requires(T t, matrix_activation_t<T> matrix_arg, fp_activation_t<T> fp_arg) {
    {t.f(matrix_arg)} -> std::same_as<matrix_activation_t<T>>;
    {t.f(fp_arg)} -> std::same_as<fp_activation_t<T>>;
    {t.f_grad(matrix_arg)} -> std::same_as<matrix_activation_t<T>>;
    {t.f_grad(fp_arg)} -> std::same_as<fp_activation_t<T>>;
};

class ReluActivation {
public:
    template <typename Derived>
    static Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>
    f(const Eigen::MatrixBase<Derived>& x) {
        return x.cwiseMax(0);
    }

    template <typename Derived>
    static Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>
    f_grad(const Eigen::MatrixBase<Derived>& x) {
        return (x.array() > 0).template cast<typename Derived::Scalar>();
    }

    static fp_t f(fp_t x) {
        return std::max(0.0f, x);
    }

    static fp_t f_grad(fp_t x) {
        if (x > 0) return 1;
        else return 0;
    }
};

class LeakyReluActivation {
public:
    static constexpr fp_t alpha = 0.01;

    template <typename Derived>
    static Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>
    f(const Eigen::MatrixBase<Derived>& x) {
        return x.unaryExpr([](typename Derived::Scalar elem) { return elem >= 0 ? elem : alpha * elem; });
    }

    template <typename Derived>
    static Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>
    f_grad(const Eigen::MatrixBase<Derived>& x) {
        return x.unaryExpr([](typename Derived::Scalar elem) { return elem > 0 ? 1 : alpha; });
    }

    static fp_t f(fp_t x) {
        return x >= 0 ? x : alpha * x;
    }

    static fp_t f_grad(fp_t x) {
        return x > 0 ? 1 : alpha;
    }
};

class SigmoidActivation {
public:
    // For Eigen::Matrix
    template <typename Derived>
    static Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>
    f(const Eigen::MatrixBase<Derived>& x) {
        return (1 / (1 + (-x.array()).exp())).matrix();
    }

    // Gradient of Sigmoid for Eigen::Matrix
    template <typename Derived>
    static Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>
    f_grad(const Eigen::MatrixBase<Derived>& x) {
        Eigen::Array<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime> sigmoid_array = f(x).array();
        return (sigmoid_array * (1 - sigmoid_array)).matrix();
    }

    // For single floating-point value
    static fp_t f(fp_t x) {
        return 1 / (1 + std::exp(-x));
    }

    // Gradient of Sigmoid for single floating-point value
    static fp_t f_grad(fp_t x) {
        double sigmoid_value = f(x);
        return sigmoid_value * (1 - sigmoid_value);
    }
};


template <>
struct activation_traits<ReluActivation> {
    using matrix_type = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;
    using fp_type = fp_t;
};

template <>
struct activation_traits<SigmoidActivation> {
    using matrix_type = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;
    using fp_type = fp_t;
};

template <>
struct activation_traits<LeakyReluActivation> {
    using matrix_type = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;
    using fp_type = fp_t;
};
#endif //CPP_NEURON_ACTIVATION_H
