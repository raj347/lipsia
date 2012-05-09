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

static void print_nothing(const char *s)
{

}


static int rand_int(int n) {
  int limit = RAND_MAX - RAND_MAX % n;
  int rnd;
  
  do {
    rnd = rand();
  } while (rnd >= limit);
  return rnd % n;
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
  printConfiguration();
}

MriSvm::~MriSvm() {

}

void MriSvm::shuffle(int *array) {
  int j,tmp;
  
  for (int sample_index(number_of_samples_ - 1); sample_index > 0; sample_index--) {
    j = rand_int(sample_index + 1);
    tmp = array[j];
    array[j] = array[sample_index];
    array[sample_index] = tmp;
  }
  
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

double MriSvm::cross_validate(int count, int leaveout) {
  int number_of_trainings_samples = number_of_samples_ - leaveout;
  
  /* Sanity Checks */
  if (number_of_trainings_samples < 2) {
    cerr << "I need at least two trainings samples in cross validation." << endl;
    exit(-1);
  }
  if (count < 1) {
    cerr << "I need at least two cross validation rounds." << endl;
    exit(-1);
  }
  
  /* Seed random number generator */
  srand(time(NULL));
  /* Don't let libsvm print anything out */
  svm_set_print_string_function(print_nothing);
  
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
  
  /* Convert all data to libsvm-Format */
  struct svm_node all_data[number_of_samples_];
  for (int sample_index(0); sample_index < number_of_samples_; sample_index++) {
    all_data[sample_index].values  = (double *) calloc(number_of_features_,sizeof(double));
    all_data[sample_index].dim     = number_of_features_;
    for(long int feature_index(0); feature_index < number_of_features_; feature_index++) {
      all_data[sample_index].values[feature_index] = sample_features_[sample_index][feature_index];
    }
  }
  
  /* Training Problem (will be a sub problem of all_data) */
  double training_classes[number_of_trainings_samples];
  struct svm_problem trainings_problem;
  trainings_problem.l = number_of_trainings_samples;
  trainings_problem.x = (struct svm_node *) calloc(number_of_trainings_samples,sizeof(struct svm_node));
  
  /* Array containing the indices of samples */
  int shuffle_indices[number_of_samples_];
  for (int sample_index(0); sample_index < number_of_samples_; sample_index++) {
    shuffle_indices[sample_index] = sample_index;
  }

  int total_correct = 0;
  for(int cross_validation_loop(0); cross_validation_loop < count; cross_validation_loop++) {
    shuffle(shuffle_indices);
   
    // Train SVM
    for(int trainings_index(0); trainings_index < number_of_trainings_samples;trainings_index++) {
      int original_index = shuffle_indices[trainings_index];
      training_classes[trainings_index]           = classes_[original_index];
      trainings_problem.x[trainings_index].values = all_data[original_index].values;
      trainings_problem.x[trainings_index].dim    = number_of_features_;
    }
    
    trainings_problem.y = training_classes;
    struct svm_model *trained_model = svm_train(&trainings_problem,&parameters_);
    
    // Predict
    for (int prediction_index = 0; prediction_index < leaveout; prediction_index++) {
      int original_index = shuffle_indices[number_of_trainings_samples + prediction_index];
      double prediction = svm_predict(trained_model,&(all_data[original_index]));
      if (prediction == classes_[original_index]) {
        total_correct++;
      }
    }
    
  }
  return (double) total_correct / (double) (count * leaveout);
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

