#include <iostream>
#include "ukf.h"

UKF::UKF() {
  //TODO: Auto-generated constructor stub
  Init();
}

UKF::~UKF() {
  // TODO: Auto-generated destructor stub
}

void UKF::Init() {

}

/*
** Programming assignment functions:
*/

void UKF::GenerateSigmaPoints(MatrixXd* Xsig_out) {
  // set state dimension
  int n_x = 5;
  // define spreading parameter
  double lambda = 3 - n_x;
  // set example state
  VectorXd x = VectorXd(n_x);
  x << 5.7441,
       1.3800,
       2.2049,
       0.5015,
       0.3528;
  // set example covariance matrix
  MatrixXd P = MatrixXd(n_x, n_x);
  P <<  0.0043, -0.0013, 0.0030, -0.0022, -0.0020,
       -0.0013,  0.0077, 0.0011,  0.0071,  0.0060,
        0.0030,  0.0011, 0.0054,  0.0007,  0.0008,
       -0.0022,  0.0071, 0.0007,  0.0098,  0.0100,
       -0.0020,  0.0060, 0.0008,  0.0100,  0.0123;
  // Create sigma points matrix
  MatrixXd Xsig = MatrixXd(n_x, 2 * n_x + 1);
  // Calculate square root of P
  MatrixXd A = P.llt().matrixL();
  /*
  ** Studene part begin
  */
  // Set first column of sigma point matrix
  Xsig.col(0) = x;
  // Set sigma points as columns of matrix Xsig
  for (int i = 0; i < n_x; ++i) {
    Xsig.col(i+1)     = x + sqrt(lambda + n_x) * A.col(i);
    Xsig.col(i+1+n_x) = x - sqrt(lambda + n_x) * A.col(i);
  }

  // print result
  // std::cout << "Xsig = " << std::endl << Xsig << std::endl;
  // Write result
  *Xsig_out = Xsig; 
}

void UKF::AugmentedSigmaPoints(MatrixXd* Xsig_out) {
  // set state dimension
  int n_x = 5;
  // set augumented dimension
  int n_aug = 7;
  // process noise standard deviation longitudinal acceleration in m/s^2
  double std_a = 0.2;
  // process noise standard deviation yaw acceleration in rad/s^2
  double std_yawdd = 0.2;
  // define spreading parameter
  double lambda = 3 - n_aug;
  // set example state
  VectorXd x = VectorXd(n_x);
  x << 5.7441,
       1.3800,
       2.2049,
       0.5015,
       0.3528;
  // set example covariance matrix
  MatrixXd P = MatrixXd(n_x, n_x);
  P <<  0.0043, -0.0013, 0.0030, -0.0022, -0.0020,
       -0.0013,  0.0077, 0.0011,  0.0071,  0.0060,
        0.0030,  0.0011, 0.0054,  0.0007,  0.0008,
       -0.0022,  0.0071, 0.0007,  0.0098,  0.0100,
       -0.0020,  0.0060, 0.0008,  0.0100,  0.0123;
  // create augmented mean vector
  VectorXd x_aug = VectorXd(7);
  // create augmented state covariance
  MatrixXd P_aug = MatrixXd(7, 7);
  // create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug, 2 * n_aug + 1);

  // Student part begin

  // create augmented mean state
  x_aug.head(5) = x;
  x_aug(5) = 0;
  x_aug(6) = 0;
  // create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5, 5) = P;
  P_aug(5, 5) = std_a * std_a;
  P_aug(6, 6) = std_yawdd * std_yawdd;
  // create square root matrix
  MatrixXd L = P_aug.llt().matrixL();
  // create augmented sigma points
  Xsig_aug.col(0) = x_aug;
  for (int i = 0; i < n_aug; ++i) {
    Xsig_aug.col(i+1)       = x_aug + sqrt(lambda + n_aug) * L.col(i);
    Xsig_aug.col(i+1+n_aug) = x_aug - sqrt(lambda + n_aug) * L.col(i);
  }

  // print result
  std::cout << "Xsig_aug = " << std::endl << Xsig_aug << std::endl;
  // write result
  *Xsig_out = Xsig_aug;
}

void UKF::SigmaPointPrediction(MatrixXd* Xsig_out) {

  // set state dimension
  int n_x = 5;
  // set augmented dimension
  int n_aug = 7;
  // create example sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug, 2 * n_aug + 1);
  Xsig_aug << 
     5.7441,  5.85768,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.63052,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,
      1.38,  1.34566,  1.52806,     1.38,     1.38,     1.38,     1.38,     1.38,   1.41434,  1.23194,     1.38,     1.38,     1.38,     1.38,     1.38,
    2.2049,  2.28414,  2.24557,  2.29582,   2.2049,   2.2049,   2.2049,   2.2049,   2.12566,  2.16423,  2.11398,   2.2049,   2.2049,   2.2049,   2.2049,
    0.5015,  0.44339, 0.631886, 0.516923, 0.595227,   0.5015,   0.5015,   0.5015,   0.55961, 0.371114, 0.486077, 0.407773,   0.5015,   0.5015,   0.5015,
    0.3528, 0.299973, 0.462123, 0.376339,  0.48417, 0.418721,   0.3528,   0.3528,  0.405627, 0.243477, 0.329261,  0.22143, 0.286879,   0.3528,   0.3528,
         0,        0,        0,        0,        0,        0,  0.34641,        0,         0,        0,        0,        0,        0, -0.34641,        0,
         0,        0,        0,        0,        0,        0,        0,  0.34641,         0,        0,        0,        0,        0,        0, -0.34641;

  // create matrix with predicted sigma points as columns
  MatrixXd Xsig_pred = MatrixXd(n_x, 2 * n_aug + 1);
  double delta_t = 0.1; // time diff in second

  // predict sigma points
  for (int i = 0; i < 2 * n_aug + 1; ++i) {
    //extract values for better readability
    double p_x = Xsig_aug(0, i);
    double p_y = Xsig_aug(1, i);
    double v   = Xsig_aug(2, i);
    double yaw = Xsig_aug(3, i);
    double yawd = Xsig_aug(4, i);
    double nu_a = Xsig_aug(5, i);
    double nu_yawdd = Xsig_aug(6, i);
    // predicted state values
    double px_p, py_p;
    // avoid division by zero
    if (fabs(yawd) > 0.001) {
      px_p = p_x + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
      py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
    }
    else {
      px_p = p_x + v * delta_t * cos(yaw);
      py_p = p_y + v * delta_t * sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd * delta_t;
    double yawd_p = yawd;

    // add noise
    px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
    py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
    v_p  = v_p  + nu_a * delta_t;

    yaw_p = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
    yawd_p = yawd_p + nu_yawdd * delta_t;

    //write predicted sigma points into right column
    Xsig_pred(0, i) = px_p;
    Xsig_pred(1, i) = py_p;
    Xsig_pred(2, i) = v_p;
    Xsig_pred(3, i) = yaw_p;
    Xsig_pred(4, i) = yawd_p;
  }


  // print result
  std::cout << "Xsig_pred = " << std::endl << Xsig_pred << std::endl;
  // write result
  *Xsig_out = Xsig_pred;

}