#include <iostream>
#include <vector>
#include <math.h>

static float normpdf(float x, float mu, float std);
std::vector<float> initialize_priors(int map_size, std::vector<float> landmark_positions,
                   float position_stdev);
float motion_model(float pseudo_position, float movement, std::vector<float> priors,
                   int map_size, int control_stdev);

int main() {

  // Set standard deviation of control
  float control_stdev = 1.0f;
  // set standard deviation of position
  float position_stdev = 1.0f;
  // meters vehicle moves per time step
  float movement_per_timestep = 1.0f;
  // number of x positions on map
  int map_size = 25;
  // initialize landmarks
  std::vector<float> landmark_positions {5, 10, 20};
  // initialize priors
  std::vector<float> priors = initialize_priors(map_size, landmark_positions,
                                                position_stdev);
  // step through each pseudo position x(i)
  for (unsigned int i = 0; i < map_size; ++i) {
    float pseudo_position = float(i);
    // get the motion model probability for each x position
    float motion_prob = motion_model(pseudo_position, movement_per_timestep, 
                                     priors, map_size, control_stdev);
    // print the result
    std::cout << pseudo_position << "\t" << motion_prob << "\n";
  }
  return 0;
}

// implement the normalized probability distribution function.
static float normpdf(float x, float mu, float std) {
  return (1/(sqrt(2*M_PI)*std))*exp(-0.5*pow((x-mu)/std, 2));
}

// implement the motion model, calculate prob of being at an estimated postion at time t
float motion_model(float pseudo_position, float movement, std::vector<float> priors,
                   int map_size, int control_stdev) {
  // initialize probability
  float position_prob = 0.0f;

  // YOUR CODE HERE
  // loop over state space for all possible positions x (convolution)
  for (int j = 0; j < map_size; ++j) {
    float next_pseudo_position =  float(j);
    // distance from i to j
    float distance_ij = pseudo_position - next_pseudo_position;

    // transition probabilities:
    float transition_prob = normpdf(distance_ij, movement, control_stdev);
    // estimate probability for the motion model
    position_prob += transition_prob * priors[j];
  }
  return position_prob;
}

// initialize priors assuming vehicle at landmark +/- 1.0 meters position stdev
std::vector<float> initialize_priors(int map_size, std::vector<float> landmark_positions,
                   float position_stdev) {
  // set all priors to 0.0
  std::vector<float> priors(map_size, 0.0);
  // set priors for each possible landmark position
  float normalization_term = landmark_positions.size() * (position_stdev * 2 + 1);
  for (unsigned int i = 0; i < landmark_positions.size(); ++i) {
    int landmark_center = landmark_positions[i];
    priors[landmark_center] = 1.0f / normalization_term;
    priors[landmark_center - 1] = 1.0f / normalization_term;
    priors[landmark_center + 1] = 1.0f / normalization_term;
  }
  return priors;
}