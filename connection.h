//
// Created by David Daeschler on 4/7/23.
//

#ifndef CPP_NEURON_CONNECTION_H
#define CPP_NEURON_CONNECTION_H

#include "common.h"
#include <vector>
#include <functional>

template<floating_point SigType>
class Connection {
private:
    decltype(1.0) weight_;

    using OutputFunction = std::function<void(SigType)>;
    std::vector<OutputFunction> outputs_;

public:
    explicit Connection(floating_point auto weight = 1.0) : weight_(weight) {}

    constexpr auto get_weight() const noexcept -> decltype(weight_) { return weight_; }
    constexpr void set_weight(floating_point auto new_weight) noexcept { weight_ = new_weight; }

    void signal(floating_point auto sig) {
        for (auto& output : outputs_) {
            output(sig * weight_);
        }
    }

    void addOutput(OutputFunction newOutput) {
        outputs_.push_back(newOutput);
    }
};

#endif //CPP_NEURON_CONNECTION_H
