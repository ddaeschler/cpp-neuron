//
// Created by David Daeschler on 4/19/23.
//

#ifndef CPP_NEURON_LAYER_H
#define CPP_NEURON_LAYER_H

#include "common.h"
#include "activation.h"
#include "initialization.h"

#include <Eigen/Dense>
#include <functional>
#include <tuple>
#include <iostream>

template<int InputCount, int OutputCount, typename Activation>
requires HasActivation<Activation>
class Layer {
public:
    typedef Eigen::Matrix<fp_t, OutputCount, InputCount> LayerMatrix;
    typedef Eigen::Vector<fp_t, OutputCount> BiasMatrix;
    typedef Eigen::Vector<fp_t, OutputCount> Outputs;

    template <typename M>
    struct FwdOutput {
        M unactivated;
        M activated;

        friend std::ostream &operator<<(std::ostream &os, const FwdOutput<M>& fo) {
            os << "unactivated: \n" << fo.unactivated << "\n\n"
               << "activated: \n" << fo.activated;

            return os;
        }
    };

    struct BackOutput {
        LayerMatrix oldWeights;
        Outputs activated;
    };

private:
    Activation activation_;
    LayerMatrix weights_;
    BiasMatrix biases_;

public:
    explicit Layer(const Activation& activation) : activation_(activation) {
        Initialization::WeightInitHE(weights_);
        biases_.setZero();
    }

    explicit Layer(const Activation& activation, const LayerMatrix& weights, const BiasMatrix& biases)
    : activation_(activation), weights_(weights), biases_(biases) {
    }

    template <typename T>
    auto forward(const T& inputs) const {
        auto out = (weights_ * inputs) + biases_;
        auto out_activated = activation_.f(out);

        using ResultType = typename Eigen::internal::remove_all<decltype(out)>::type;
        using Traits = typename Eigen::internal::traits<ResultType>;

        return FwdOutput<Eigen::Matrix<fp_t, Traits::RowsAtCompileTime, Traits::ColsAtCompileTime>> { out, out_activated };
    }

    template <typename Error, typename SelfOutUnactivated, typename LayerInputs>
    auto backProp(const Error& e, const SelfOutUnactivated& selfOut, const LayerInputs& inputs, fp_t learningRate=1e-4f) {
        auto delta = activation_.f_grad(selfOut).cwiseProduct(e);
        auto oldWeights = weights_;
        auto weightAdj = learningRate * delta * inputs.transpose();
        auto biasAdj = learningRate * delta;
        weights_ -= weightAdj;
        biases_ -= biasAdj;

        return std::make_tuple(delta, oldWeights.transpose());
    }
};

#endif //CPP_NEURON_LAYER_H
