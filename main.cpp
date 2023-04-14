#include <Eigen/Dense>

#include <iostream>
#include <algorithm>

typedef float fp_t;


template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>
relu(const Eigen::MatrixBase<Derived>& x) {
    return x.cwiseMax(0);
}

fp_t relu(fp_t x) {
    return std::max(0.0f, x);
}

using std::cout;
using std::endl;

int main() {


    // our inputs
    Eigen::Matrix<fp_t, 3, 1> inputs; inputs <<
        1.0f,
        2.0f,
        3.0f;

    // for layers, each matrix contains the weights between layers
    // I = input layer, H = hidden layer, O = output layer
    // w = weight, b = bias

    // Input to Hidden
    Eigen::Matrix<fp_t, 64, 3> wItoH;
    wItoH.setRandom();
    Eigen::Vector<fp_t, 64> bItoH;
    bItoH.setZero();

    // Hidden to output
    Eigen::Matrix<fp_t, 1, 64> wHtoO;
    wHtoO.setRandom();
    fp_t bHtoO = 0;


    //take a step forward and record intermediates
    Eigen::Vector<fp_t, 64> ih_out = relu((wItoH * inputs) + bItoH);
    float price = relu((wHtoO * ih_out) + bHtoO);

    cout << price << endl;

    //backprop
    


    return 0;
}
