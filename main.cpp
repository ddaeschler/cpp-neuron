#include <Eigen/Dense>

#include <iostream>
#include <algorithm>

#include "layer.h"
#include "activation.h"

using std::cout;
using std::endl;

int main() {
    // our inputs
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
    Layer<inputs.rows(), 5, ReluActivation> ih(a);
    Layer<5, 5, ReluActivation> ihH1(a);
    Layer<5, 1, ReluActivation> ho(a);

    for (int i = 0; i < 5000; i++) {
        auto ihOut = ih.forward(inputs);
        auto ihH1Out = ihH1.forward(ihOut.activated);
        auto hoOut = ho.forward(ihH1Out.activated);

        if (hoOut.unactivated[0] < 0.0f) {
            //we're dead. re-init
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




    return 0;
}
