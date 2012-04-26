/**
 * MriSvm.cpp
 *
 *  Created on: 03.04.2012
 *      Author: Tilo Buschmann
 */

#include <iostream>
#include <fstream>

#include "MriSvm.h"

using std::ofstream;
using std::cerr;
using std::cout;
using std::endl;

static void print_nothing(const char *s)
{

}

MriSvm::MriSvm( long int number_of_samples,
                long int number_of_features, 
                sample_features_array_type sample_features, 
                vector<int> classes
               ) : 
                number_of_samples_(number_of_samples), 
                number_of_features_(number_of_features), 
                sample_features_(sample_features), 
                classes_(classes) 
{
  parameters_ = getDefaultParameters();
  //printConfiguration();
}

MriSvm::MriSvm( long int number_of_samples,
                long int number_of_features, 
                sample_features_array_type sample_features, 
                vector<int> classes,
                int svm_type,
                int kernel_type
               ) : 
                number_of_samples_(number_of_samples), 
                number_of_features_(number_of_features), 
                sample_features_(sample_features), 
                classes_(classes) 
{
  parameters_ = getDefaultParameters();
  parameters_.svm_type    = svm_type;
  parameters_.kernel_type = kernel_type;
  //printConfiguration();
}

MriSvm::~MriSvm() {

}

void MriSvm::printConfiguration() {
  cout << "MriSvm configuration" << endl;
  cout << "number_of_samples="  << number_of_samples_ << endl;
  cout << "number_of_features=" << number_of_features_ << endl;
}

void MriSvm::scale() {
  for(long int feature_index(0); feature_index < number_of_features_; feature_index++) {
    // Find range of values of this feature
    double max = DBL_MIN;
    double min = DBL_MAX;
    for (long int sample_index(0); sample_index < number_of_samples_; sample_index++) {
      if (sample_features_[sample_index][feature_index] < min) {
        min = sample_features_[sample_index][feature_index];
      }
      if (sample_features_[sample_index][feature_index] > max) {
        max = sample_features_[sample_index][feature_index];
      }
    }

    // Scale
    for (long int sample_index(0); sample_index < number_of_samples_; sample_index++) {
      double x = sample_features_[sample_index][feature_index];
      double lower = DEFAULT_MRISVM_SCALE_LOWER;
      double upper = DEFAULT_MRISVM_SCALE_UPPER;

      sample_features_[sample_index][feature_index] = lower + (upper - lower) * (x - min) / (max - min);
    }
  }
}

double MriSvm::cross_validate() {
  /* Fill response vector with classes */ 
  double y[number_of_samples_]; // Classes
  for (long int sample_index(0); sample_index < number_of_samples_; sample_index++) {
    y[sample_index] = classes_[sample_index];
  }

  /* Fill predictor matrix with voxel data */
  struct svm_problem problem;
  problem.l = number_of_samples_;
  problem.y = y;
  problem.x = (struct svm_node *) calloc(number_of_samples_,sizeof(struct svm_node));
  for (long int sample_index(0); sample_index < number_of_samples_; sample_index++) {
    problem.x[sample_index].values  = (double *) calloc(number_of_features_,sizeof(double));
    problem.x[sample_index].dim     = number_of_features_;
    for(long int feature_index(0); feature_index < number_of_features_; feature_index++) {
      problem.x[sample_index].values[feature_index] = sample_features_[sample_index][feature_index];
    }
  }
    
  // Cross validate
  double target[number_of_samples_]; 

  svm_set_print_string_function(print_nothing);
  /*
  const char *error_msg = svm_check_parameter(&problem,&parameters_);
  if(error_msg)
  {
    cerr << "ERROR: " << error_msg << endl;
    exit(1);
  }
  */
  svm_cross_validation(&problem,&parameters_,DEFAULT_MRISVM_CROSS_VALIDATION,target);

  int total_correct = 0;
  for(int i=0;i<number_of_samples_;i++) {
    if(target[i] == problem.y[i])
        ++total_correct;
  }

  // Free memory again
  for (long int sample_index(0); sample_index < number_of_samples_; sample_index++) {
    free(problem.x[sample_index].values);
  }
  free(problem.x);

  return((double) total_correct/ (double) number_of_samples_);
}

/* Export for R */
void MriSvm::export_table(std::string file_name) {
  ofstream table_file;
  table_file.open(file_name.c_str());

  //Header with sample names
  table_file << "Sample";
  for(int sample_index(0); sample_index < number_of_samples_; sample_index++) {
    table_file << "\t" << sample_index;
  }
  table_file << endl;

  // Classes
  table_file << "Class";
  for(int sample_index(0); sample_index < number_of_samples_; sample_index++) {
    table_file << "\t" << classes_[sample_index];
  }
  table_file << endl;

  // Features
  for(int feature_index(0); feature_index < number_of_features_;feature_index++) {
    table_file << "Feature " << feature_index;
    for(int sample_index(0); sample_index < number_of_samples_; sample_index++) {
      table_file << "\t" << sample_features_[sample_index][feature_index];
    }
    table_file << endl;
  }
  table_file.close();
}


struct svm_parameter MriSvm::getDefaultParameters() {
  struct svm_parameter parameters;

  parameters.svm_type     = DEFAULT_MRISVM_SVM_TYPE;
  parameters.kernel_type  = DEFAULT_MRISVM_KERNEL_TYPE;
  parameters.degree       = 3;
  parameters.gamma        = 1.0/number_of_features_;  // 1/num_features
  parameters.coef0        = 0;
  parameters.nu           = 0.5;
  parameters.cache_size   = 100;
  parameters.C            = 1;
  parameters.eps          = 1e-3;
  parameters.p            = 0.1;
  parameters.shrinking    = 1;
  parameters.probability  = 0;
  parameters.nr_weight    = 0;
  parameters.weight_label = NULL;
  parameters.weight       = NULL;

  return(parameters);

}

