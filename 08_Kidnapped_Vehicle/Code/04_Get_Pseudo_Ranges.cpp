#include <iostream>
#include <algorithm>
#include <vector>

// set standard deviation of control
float control_stdev = 1.0f;
// meters vehicle moves per time step
float movement_per_timestep = 1.0f;
// number of x positions on map
int map_size = 25;
// define landmarks
std::vector<float> landmark_positions{5, 10, 12, 20};

std::vector<float> pseudo_range_estimator(std::vector<float> landmark_positions, float pseudo_position);

int main() {
  // step through each pseudo position x(i)
  for (unsigned int i = 0; i < map_size; ++i) {
    float pseudo_position = float(i);
    // get pseudo ranges
    std::vector<float> pseudo_ranges = pseudo_range_estimator(landmark_positions, pseudo_position);
    // print to stdout
    if (pseudo_ranges.size() > 0) {
      for (unsigned int s = 0; s < pseudo_ranges.size(); ++s) {
        std::cout << "x: " << i << "\t" << pseudo_ranges[s] << std::endl;
      }
      std::cout << "-------------------------------" << std::endl;
    }
  }
  return 0;
}

// TODO: complete pseudo range estimator function
std::vector<float> pseudo_range_estimator(std::vector<float> landmark_positions, float pseudo_position) {
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