#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include "ukf.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

int main() {
  // Create a UKF instance
  UKF ukf;
  
  // Programming assignment calls
  MatrixXd Xsig_pred = MatrixXd(5, 15);
  ukf.SigmaPointPrediction(&Xsig_pred);

  // Print result
  std::cout << "Xsig_pred = " << std::endl << Xsig_pred << std::endl;

  return 0;
}