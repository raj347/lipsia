/**
 * MriSvm.cpp
 *
 *  Created on: 03.04.2012
 *      Author: Tilo Buschmann
 */

#include <iostream>
#include <fstream>

#include <boost/foreach.hpp>

#include "MriSvm.h"

using std::ofstream;
using std::cerr;
using std::cout;
using std::endl;

/*
 * To silence libsvm
 * 
 */

static void print_nothing(const char *s)
{

}

MriSvm::MriSvm(
                sample_features_array_type  sample_features, 
                vector<int>                 classes,
                long int                    number_of_samples,
                long int                    number_of_features
               ) : 
                classes_(classes),
                number_of_samples_(number_of_samples), 
                number_of_features_(number_of_features)
{
  construct(sample_features);
  //printConfiguration();
}

MriSvm::~MriSvm() {
  for (int sample_index(0); sample_index < number_of_samples_; sample_index++) {
    delete[] all_data_[sample_index].values;
  }
  
  delete[] all_data_;
}

void MriSvm::construct(sample_features_array_type  &sample_features) {
  parameters_ = get_default_parameters();
  prepare_all_data(sample_features);
}

void MriSvm::prepare_all_data(sample_features_array_type  &sample_features) {
  /* Problem looks this way (e.g. with 6 samples and 3 features):
   * .l = 6
   *       _ _ _ _ _ _
   * .y = |_|_|_|_|_|_|
   * 
   *        _
   * .x -> | |.dim = 3;   _ _ _
   *       |_|.values -> |_|_|_|
   *       | |.dim = 3;   _ _ _
   *       |_|.values -> |_|_|_|
   *       | |.dim = 3;   _ _ _
   *       |_|.values -> |_|_|_|
   *       | |.dim = 3;   _ _ _
   *       |_|.values -> |_|_|_|
   *       | |.dim = 3;   _ _ _
   *       |_|.values -> |_|_|_|
   *       | |.dim = 3;   _ _ _
   *       |_|.values -> |_|_|_|
   *       
   */
  all_data_   = new svm_node[number_of_samples_] ;
  
  /* Convert all data to libsvm-Format */
  for (int sample_index(0); sample_index < number_of_samples_; sample_index++) {
    all_data_[sample_index].values  = new double[number_of_features_];
    all_data_[sample_index].dim     = number_of_features_;
    
    for(long int feature_index(0); feature_index < number_of_features_; feature_index++) {
      all_data_[sample_index].values[feature_index] = sample_features[sample_index][feature_index];
    }
    
  }
}

void MriSvm::scale() {
  for(long int feature_index(0); feature_index < number_of_features_; feature_index++) {
    // Find range of values of this feature
    double max = DBL_MIN;
    double min = DBL_MAX;
    for (long int sample_index(0); sample_index < number_of_samples_; sample_index++) {
      double x = all_data_[sample_index].values[feature_index];
      if (x < min) {
        min = x;
      }
      if (x > max) {
        max = x;
      }
    }

    // Scale
    for (long int sample_index(0); sample_index < number_of_samples_; sample_index++) {
      double x = all_data_[sample_index].values[feature_index];
      double lower = DEFAULT_MRISVM_SCALE_LOWER;
      double upper = DEFAULT_MRISVM_SCALE_UPPER;

      all_data_[sample_index].values[feature_index] = lower + (upper - lower) * (x - min) / (max - min);
    }
  }
}

void MriSvm::set_svm_type(int svm_type) {
    parameters_.svm_type = svm_type;
}

void MriSvm::set_svm_kernel_type(int svm_kernel_type) {
    parameters_.kernel_type = svm_kernel_type;
}


void MriSvm::printConfiguration() {
  cout << "MriSvm configuration" << endl;
  cout << "number_of_samples="  << number_of_samples_ << endl;
  cout << "number_of_features=" << number_of_features_ << endl;
  
  cout << "Classes" << endl;
  cout << "=======" << endl;
  for (int classes_index(0); classes_index < number_of_samples_; classes_index++) {
    cout << classes_[classes_index] << " ";
  }
  cout << endl;
}

/**
 * 
 * @param permutation_count number of permutations
 * @param leaveout number of samples to be left out in cross validation
 * 
 * @return list of @count searchlightsvms at this position
 */

void MriSvm::Permutate(permutated_validities_type &permutated_validities, 
                          int number_of_permutations,
                          permutations_array_type &permutations,
                          int leaveout,
                          int band, 
                          int row, 
                          int column) {
  
  struct svm_node *data_base = new svm_node[number_of_samples_];
  
  for (int permutation_loop(0); permutation_loop < number_of_permutations; permutation_loop++) {
    // Set up a SVM problem with original classes but permutated samples
    for (int sample_loop(0); sample_loop < number_of_samples_; sample_loop++) {
      int shuffled_index = permutations[permutation_loop][sample_loop];
      
      data_base[sample_loop].values = all_data_[shuffled_index].values;
      data_base[sample_loop].dim    = number_of_features_;
    }
    
    // Calculate validity
    permutated_validities[permutation_loop][band][row][column] = cross_validate(leaveout,data_base);
  }
  
  delete[] data_base;
  
  return;
}

double MriSvm::cross_validate(int leaveout) {
  return cross_validate(leaveout,all_data_);
}

double MriSvm::cross_validate(int leaveout,struct svm_node *data_base) {
  /* Sanity Check */
  if ((number_of_samples_ - leaveout) < 2) {
    cerr << "I need at least two trainings samples in cross validation." << endl;
    exit(-1);
  }
  
  /* Don't let libsvm print anything out */
  svm_set_print_string_function(print_nothing);
  
  /* Training Problem (will be a sub problem of all_data) */
  double training_classes[number_of_samples_];
  struct svm_problem trainings_problem;
  trainings_problem.x = new svm_node[number_of_samples_];
  trainings_problem.y = training_classes;
  
  int total_correct = 0;
  
  // Calculate number of shuffles
  int count = number_of_samples_ / leaveout;
  if ((number_of_samples_ % leaveout) != 0)
    count++;
  
  for(int cross_validation_loop(0); cross_validation_loop < count; cross_validation_loop++) {
    volatile int trainings_index = 0;
    for(int trainings_loop(0); trainings_loop < number_of_samples_;trainings_loop++) {
      if ((trainings_loop % count) != cross_validation_loop) {
        training_classes[trainings_index]           = classes_[trainings_loop];
        trainings_problem.x[trainings_index].values = data_base[trainings_loop].values;
        trainings_problem.x[trainings_index].dim    = number_of_features_;
        trainings_index++;
      }
    }
    trainings_problem.l = trainings_index;
    
    // Train SVM
    struct svm_model *trained_model = svm_train(&trainings_problem,&parameters_);
    
    // Predict
    for (int prediction_index = cross_validation_loop; prediction_index < number_of_samples_; prediction_index += count) {
      double prediction = svm_predict(trained_model,&(data_base[prediction_index]));
      if (prediction == classes_[prediction_index]) {
        total_correct++;
      }
    }
    /* Free all that unnecessary stuff */
    svm_free_and_destroy_model(&trained_model);
  }
  delete[] trainings_problem.x;
  
  return (double) total_correct / (double) (number_of_samples_);
}

struct svm_parameter MriSvm::get_default_parameters() {
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

