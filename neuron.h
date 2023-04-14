//
// Created by David Daeschler on 4/7/23.
//

#ifndef CPP_NEURON_NEURON_H
#define CPP_NEURON_NEURON_H

#include "common.h"
#include <functional>
#include <string>
#include <iostream>

template<floating_point SigType>
class Neuron {
private:
    SigType current_sum_;
    int input_count_;
    int sigs_received_;

    using ActivationFunction = std::function<SigType(SigType)>;
    using OutputFunction = std::function<void(SigType)>;
    ActivationFunction activation_fn_;
    std::string name_;

    std::vector<OutputFunction> outputs_;

private:
    void reset() {
        current_sum_ = 0;
        sigs_received_ = 0;
    }

public:
    explicit Neuron(ActivationFunction fn, int inputCount, std::string name)
        : current_sum_(0), input_count_(inputCount), sigs_received_(0), activation_fn_(fn), name_(name) {}

    void signal(SigType sig) {
        current_sum_ += sig;

        //std::cout << "Neuron(" << name_ << ") got " << sig << std::endl;

        if (++sigs_received_ == input_count_) {
            // pulse the downstream connections
            SigType result = activation_fn_(current_sum_);

            std::cout << "Neuron(" << name_ << ") sending " << result << std::endl;

            for (auto output : outputs_) {
                output(result);
            }

            reset();
        }
    }

    void addOutput(OutputFunction newOutput) {
        outputs_.push_back(newOutput);
    }
};


#endif //CPP_NEURON_NEURON_H
