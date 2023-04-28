#include <Eigen/Dense>

#include <iostream>
#include <algorithm>

#include "common.h"

#include "layer.h"
#include "activation.h"


template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>
relu(const Eigen::MatrixBase<Derived>& x) {
    return x.cwiseMax(0);
}

template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>
relu_grad(const Eigen::MatrixBase<Derived>& x) {
    return (x.array() > 0).template cast<typename Derived::Scalar>();
}

fp_t relu(fp_t x) {
    return std::max(0.0f, x);
}

fp_t relu_grad(fp_t x) {
    if (x > 0) return 1;
    else return 0;
}

using std::cout;
using std::endl;

int main() {
    // our inputs
    Eigen::Matrix<fp_t, 5, 1> inputs; inputs <<
        1.0f,
        2.0f,
        3.0f,
        4.0f,
        5.0f;

    // expected output
    fp_t y = 5;

    // learning rate
    static const fp_t LEARNING_RATE = 1e-4;

    ReluActivation a;
    Layer<inputs.rows(), 64, ReluActivation> ih(a);
    Layer<64, 1, ReluActivation> ho(a);

    for (int i = 0; i < 300; i++) {
        auto ihOut = ih.forward(inputs);
        auto hoOut = ho.forward(ihOut.activated);

        if (hoOut.unactivated[0] < 0.0f) {
            //we're dead. dump state
            cout << "ReLU died." << endl;
            break;
        }

        cout << "res: " << hoOut.activated << endl;

        //now start with the rightmost layer and backprop
        auto delta = ho.backProp(hoOut.activated - Eigen::Vector<fp_t, 1>(y), hoOut.unactivated,
                                 ihOut.activated, LEARNING_RATE);
        ih.backProp( std::get<1>(delta) * std::get<0>(delta), ihOut.unactivated, inputs, LEARNING_RATE);
    }




    return 0;
}
