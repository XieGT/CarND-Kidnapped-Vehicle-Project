/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) 
{
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  std::default_random_engine gen;

  num_particles = 100;  // TODO: Set the number of particles

  // prepare gaussian distributions with the provided GPS x, y, theta
  std::normal_distribution<double> GPS_x(x, std[0]);
  std::normal_distribution<double> GPS_y(y, std[1]);
  std::normal_distribution<double> GPS_theta(theta,std[2]);

  for (int i = 0; i < num_particles; ++i)
  {
  	Particle particle;
  	particle.id = i;
  	particle.x = GPS_x(gen);
  	particle.y = GPS_y(gen);
  	particle.theta = GPS_theta(gen);
  	particle.weight = 1;

  	particles.push_back(particle);
  	weights.push_back(1);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) 
{
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
	std::default_random_engine gen;

  // sensor noise for prediction steps (gaussain distributions)
  std::normal_distribution<double> sensorNoise_x(0, std_pos[0]);
  std::normal_distribution<double> sensorNoise_y(0, std_pos[1]);
  std::normal_distribution<double> sensorNoise_theta(0, std_pos[2]);

  for (int j = 0; j < num_particles; ++j)
  {

  	if (fabs(yaw_rate) < 1e-3)
  	{
  		// zero yaw rate
  		particles[j].x += velocity * delta_t * cos(particles[j].theta) + sensorNoise_x(gen);
  		particles[j].y += velocity * delta_t * sin(particles[j].theta) + sensorNoise_y(gen);
      particles[j].theta += sensorNoise_theta(gen);

  	}
  	
  	else
  	{
  		// nonzero yaw rate
  		particles[j].x += (velocity/yaw_rate) * (sin(particles[j].theta + yaw_rate * delta_t) - sin(particles[j].theta)) + sensorNoise_x(gen);
  		particles[j].y += (velocity/yaw_rate) * (cos(particles[j].theta) - cos(particles[j].theta + yaw_rate * delta_t)) + sensorNoise_y(gen);
  		particles[j].theta += yaw_rate * delta_t + sensorNoise_theta(gen);
  	}
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) 
{
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
	for (int i = 0; i < observations.size(); ++i)
	{
		double min_distance = std::numeric_limits<double>::max();
		int landmark_id = -1;

		for(int j = 0; j < predicted.size(); ++j)
		{
			double cur_distance = dist(predicted[j].x, predicted[j].y, observations[i].x, observations[i].y);
			if(cur_distance < min_distance)
			{
				min_distance = cur_distance;
				landmark_id = j;
			}
		}
		observations[i].id = landmark_id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) 
{
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  // 0. Prepare the parameters for 
  double den_x = 2.0 * std_landmark[0] * std_landmark[0];
  double den_y = 2.0 * std_landmark[1] * std_landmark[1];
  double normalizer = 1.0 / (2.0 * M_PI * std_landmark[0] * std_landmark[1]);
  
  for (int i = 0; i < num_particles; ++i)
  {
  	
  	// 1. Transform observation from particle's coordinates to map's coordinates 	
  	vector<LandmarkObs> transformed_observations;
  	
  	for (int j = 0; j < observations.size(); ++j)
  	{
  		LandmarkObs trans_obs;
  		trans_obs.x = particles[i].x + (observations[j].x * cos(particles[i].theta) - observations[j].y * sin(particles[i].theta));
  		trans_obs.y = particles[i].y + (observations[j].x * sin(particles[i].theta) + observations[j].y * cos(particles[i].theta));
  		transformed_observations.push_back(trans_obs);
  	}
  	
  	// 2. Select landmarks within a reach of particle's sensors 	
  	vector<LandmarkObs> selected_landmarks;
  	
  	for (int k = 0; k < map_landmarks.landmark_list.size(); ++k)
  	{
  		double det_distance = dist(particles[i].x, particles[i].y, map_landmarks.landmark_list[k].x_f, map_landmarks.landmark_list[k].y_f);
  		
  		if (det_distance <= sensor_range)
  		{
  			LandmarkObs landmark;
  			landmark.id = map_landmarks.landmark_list[k].id_i;
  			landmark.x = map_landmarks.landmark_list[k].x_f;
  			landmark.y = map_landmarks.landmark_list[k].y_f;
  			selected_landmarks.push_back(landmark);
  		}
  	}

  // 3. Associate selected landmarks with transformed observations	
  	dataAssociation(selected_landmarks, transformed_observations);
  
  // 4. Compute particle weight	

    double weight = 1;

  	for (int l = 0; l < transformed_observations.size(); ++l)
  	{
  		if (transformed_observations[l].id >= 0)
  		{
  			
  			double diff_x = transformed_observations[l].x - selected_landmarks[transformed_observations[l].id].x;
  			double diff_y = transformed_observations[l].y - selected_landmarks[transformed_observations[l].id].y;
  			
  			long double multipler = normalizer * exp(- diff_x*diff_x/den_x - diff_y*diff_y/den_y);
  			weight *= multipler;
      }

  	}
    particles[i].weight = weight;
  	weights[i] = particles[i].weight;	
  }
}

void ParticleFilter::resample() 
{
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
	std::default_random_engine gen;
	std::discrete_distribution<int> distribution(weights.begin(),weights.end());

	vector<Particle> resample_particles;

	for (int i = 0; i < num_particles; ++i)
	{
		resample_particles.push_back(particles[distribution(gen)]);
	}
	particles  = resample_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}