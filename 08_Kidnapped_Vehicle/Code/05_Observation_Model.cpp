#include <iostream>
#include <algorithm>
#include <vector>
#include <math.h>

// function to get pseudo ranges
std::vector<float> pseudo_range_estimator(std::vector<float> landmark_positions, 
                                          float pseudo_position);
// observation model: calculates likelihood prob term based on landmark proximity
float observation_model(std::vector<float> landmark_positions, std::vector<float> observations,
                        std::vector<float> pseudo_ranges, float distance_max, 
                        float observation_stdev);
// norm_pdf function 
static float normpdf(float x, float mu, float std);

int main() {
  // set observation standard deviation
  float observation_stdev = 1.0f;
  // number of x positions on map
  int map_size = 25;
  // set distance max
  float distance_max = map_size;
  // define landmarks
  std::vector<float> landmark_positions {5, 10, 12, 20};
  // define observations 
  std::vector<float> observations {5.5, 13, 15};
  // step through each pseudo position x(i)
  for (unsigned int i = 0; i < map_size; ++i) {
    float pseudo_position = float(i);
    // get pseudo ranges
    std::vector<float> pseudo_ranges = pseudo_range_estimator(landmark_positions, pseudo_position);
    // get observation probability
    float observation_prob = observation_model(landmark_positions, observations,
                                               pseudo_ranges, distance_max, observation_stdev);
    // print to stdout
    std::cout << observation_prob << std::endl;
  }
  return 0;
}

// TODO: complete the observation model function
// calculates likelihood prob term based on landmark proximity
float observation_model(std::vector<float> landmark_positions, std::vector<float> observations,
                        std::vector<float> pseudo_ranges, float distance_max, 
                        float observation_stdev){ 
  float distance_prob = 1.0f;
  // YOUR CODE HERE
  // loop over current observation vector
  for (unsigned int z = 0; z < observations.size(); ++z) {
    // define min distance
    float pseudo_range_min;
    // check if distance vector exists
    if (pseudo_ranges.size() > 0) {
      // set min distance
      pseudo_range_min = pseudo_ranges[0];
      // remove this entry from pseudo_range vector
      pseudo_ranges.erase(pseudo_ranges.begin());
    } else {
      // no or negative distances: set min distance to a large number
      // Can use infinity or distance_max here
      pseudo_range_min = std::numeric_limits<const float>::infinity();
    }
    // estimate the probability for observation model, this is our likelihood
    distance_prob *= normpdf(observations[z], pseudo_range_min, observation_stdev);
  }
  return distance_prob;
}

// function to get pseudo ranges
std::vector<float> pseudo_range_estimator(std::vector<float> landmark_positions, 
                                          float pseudo_position) {
  // define pseudo observation vector:
  std::vector<float> pseudo_ranges;
  // loop over number of landmarks and estimate pseudo ranges
  for (unsigned int i = 0; i < landmark_positions.size(); ++i) {
    // determine the distance between each pseudo position x and each landmark position
    float range_i = landmark_positions[i] - pseudo_position;
    // check if distances are positive
    if (range_i > 0.0f) {
      pseudo_ranges.push_back(range_i);
    }
  }
  // sort pseudo range vector
  sort(pseudo_ranges.begin(), pseudo_ranges.end());

  return pseudo_ranges;
}

static float normpdf(float x, float mu, float std) {
  return (1/(sqrt(2*M_PI)*std))*exp(-0.5*pow((x-mu)/std, 2));
}