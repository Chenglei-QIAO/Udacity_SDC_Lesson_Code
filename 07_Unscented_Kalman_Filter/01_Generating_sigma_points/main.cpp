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
  MatrixXd Xsig = MatrixXd(5, 11);
  ukf.GenerateSigmaPoints(&Xsig);

  // Print result
  std::cout << "Xsig = " << std::endl << Xsig << std::endl;

  return 0;
}