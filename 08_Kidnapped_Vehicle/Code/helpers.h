#ifndef HELP_FUNCTION_H_
#define HELP_FUNCTION_H_

#include <math.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <iomanip>

using namespace std;

class Helpers {
 public:
  // definition of one over square root of 2*pi
  constexpr static float STATIC_ONE_OVER_SQRT_2PI = 1 / sqrt(2 * M_PI);
  float ONE_OVER_SQRT_2PI = 1 / sqrt(2 * M_PI);
  /********************************************************************
  ** normpdf(X, mu, sigma) computes the probability function at valuels x using the 
  ** normal distribution with mean mu and standard deviation std. x,mu and sigma 
  ** musst be scalar! The parameter std must be positive.
  ** The normal pdf is y=f(x,mu,std) = 1/(std*sqrt(2pi))e[e(x-mu)^2/s*std^2]
  ***************************************************************************/
  static float normpdf(float x, float mu, float std) {
    return (STATIC_ONE_OVER_SQRT_2PI / std) * exp(-0.5 * pow((x - mu) / std, 2));
  }

  // static function to normalized a vector
  static std::vector<float> normalize_vector(std::vector<float> inputVector) {
    // declare sum
    float sum = 0.0f;
    // declare and resize output vector
    std::vector<float> outputVector;
    outputVector.resize(inputVector.size());
    // eatimate the sum
    for (unsigned int i = 0; i < inputVector.size(); ++i) {
      sum += inputVector[i];
    }
    // normalize with sum
    for (unsigned int i = 0; i < inputVector.size(); ++i) {
      outputVector[i] = inputVector[i] / sum;
    }
    // return normalized vector
    return outputVector;
  }
};

#endif /* HELP_FUNCTION_H_ */