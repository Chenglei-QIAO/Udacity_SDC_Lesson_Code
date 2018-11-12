#include <iostream>
#include <vector>
#include <math.h>

static float normpdf(float x, float mu, float std);
// Assign a value that maximizes the returned probabilities of norm_pdf
float value = 2; 
// set a control parameter or observattion measurement
float parameter = 1.0; 
// position or observattion standard deviation
float stdev = 1.0;

int main() {
  // Calculate the probability
  float prob = normpdf(value, parameter, stdev);
  // print the result
  std::cout << "Probability: " << prob << "\n";

  return 0;
}

static float normpdf(float x, float mu, float std) {
  return (1/(sqrt(2*M_PI)*std))*exp(-0.5*pow((x-mu)/std, 2));
}