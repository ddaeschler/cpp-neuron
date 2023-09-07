#include <Eigen/Dense>

#include <iostream>
#include <algorithm>

#include "linear_layer.h"
#include "activation.h"

using std::cout;
using std::endl;

void simple_regression() {
    // our rows
    Eigen::Matrix<fp_t, 5, 1> inputs; inputs <<
                                             0.3f,
            0.4f,
            0.5f,
            0.6f,
            0.7f;

    // expected output
    fp_t y = 0.9;

    // learning rate
    static const fp_t LEARNING_RATE = 1e-3;

    ReluActivation a;
    LinearLayer<inputs.rows(), 3, ReluActivation> ih(a);
    LinearLayer<3, 3, ReluActivation> ihH1(a);
    LinearLayer<3, 1, ReluActivation> ho(a);

    for (int i = 0; i < 5000; i++) {
        auto ihOut = ih.forward(inputs);
        auto ihH1Out = ihH1.forward(ihOut.activated);
        auto hoOut = ho.forward(ihH1Out.activated);

        if (hoOut.unactivated[0] < 0.0f) {
            //we're dead. re-init. this shouldn't happen with real data
            ih.initialize();
            ihH1.initialize();
            ho.initialize();
            continue;
        }

        cout << "res: " << hoOut.activated << endl;

        //now start with the rightmost layer and backprop
        auto delta1 = ho.backProp(hoOut.activated - Eigen::Vector<fp_t, 1>(y), hoOut.unactivated,
                                  ihH1Out.activated, LEARNING_RATE);
        auto delta2 = ihH1.backProp( std::get<1>(delta1) * std::get<0>(delta1), ihH1Out.unactivated,
                                     ihOut.activated, LEARNING_RATE);
        ih.backProp( std::get<1>(delta2) * std::get<0>(delta2), ihOut.unactivated, inputs, LEARNING_RATE);
    }
}

void simple_classification() {
    // our rows
    Eigen::Matrix<fp_t, 3, 3> inputs; inputs <<
        0.1f, 0.2f, 0.3f, // dog
        0.2f, 0.1f, 0.3f, // cat
        0.3f, 0.1f, 0.2f // pony
        ;

    // expected outputs. indicates which row output should be hot
    const int DOG = 0;
    const int CAT = 1;
    const int PONY = 2;

    // learning rate
    static const fp_t LEARNING_RATE = 1e-3;

    // build the network

}

int main() {
    simple_regression();
    simple_classification();

    return 0;
}
