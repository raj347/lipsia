/**
 * @file MriSvm.cpp
 *
 *  Created on: 03.04.2012
 *  
 * @author Tilo Buschmann
 * 
 */

#include <iostream>
#include <fstream>

#include <boost/foreach.hpp>

// GSL headers
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>

#include "MriSvm.h"

using std::ofstream;
using std::cerr;
using std::cout;
using std::endl;

/**
 * A function that does nothing. Used as logging-callback function for libsvm.
 * 
 * @param[in] s a c-string, not to be printed
 * 
 */
static void print_nothing(const char *s)
{

}

/**
 * Round value
 * 
 * @param[in] r value to be rounded
 * 
 * @return rounded value
 */
double round(double r) {
    return (r > 0.0) ? floor(r + 0.5) : ceil(r - 0.5);
}


/**
 * MriSVM implements a connection to libsvm
 * 
 * @param[in] sample_features array containing samples and features
 * @param[in] classes vector containing the classes of samples
 * @param[in] number_of_samples number of samples
 * @param[in] number_of_features number of features
 * 
 */
MriSvm::MriSvm(
                sample_features_array_type  sample_features, 
                vector<double>              classes,
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

/**
 * Destructor that deletes prepared libsvm-data
 * 
 */
MriSvm::~MriSvm() {
  for (int sample_index(0); sample_index < number_of_samples_; sample_index++) {
    delete[] all_data_[sample_index].values;
  }
  
  delete[] all_data_;
}

/**
 * Initialisation at construction, sets defaults parameters and converts data to libsvm format
 * 
 * @param[in] sample_features input data
 */
void MriSvm::construct(sample_features_array_type  &sample_features) {
  parameters_ = get_default_parameters();
  parameters_.gamma = 1 / number_of_samples_;
  prepare_all_data(sample_features);
}

/**
 * Prepare input data for use with libsvm
 * 
 * @param[in] sample_features input data
 */
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

/**
 * Rescale data to be between -1 and 1 (default)
 */

void MriSvm::scale() {
  for(long int feature_index(0); feature_index < number_of_features_; feature_index++) {
    // Find mean
    double sum = 0.0;
    for (long int sample_index(0); sample_index < number_of_samples_; sample_index++) {
      sum += all_data_[sample_index].values[feature_index];
    }
    double mean = sum/number_of_samples_;
    // Calculate standard deviation
    sum = 0.0;
    for (long int sample_index(0); sample_index < number_of_samples_; sample_index++) {
      double diff = all_data_[sample_index].values[feature_index] - mean;
      sum += diff * diff;
    }

    double sd = sqrt(sum / (number_of_samples_-1));

    // center and scale
    for (long int sample_index(0); sample_index < number_of_samples_; sample_index++) {
      all_data_[sample_index].values[feature_index] -= mean;
      all_data_[sample_index].values[feature_index] /= sd;
    }

    /*
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
    */
  }
}

/**
 * Set SVM TYPE to be used
 * 
 * @param[in] svm_type  svm type to be used (as defined by libsvm)
 * 
 */
void MriSvm::set_svm_type(int svm_type) {
    parameters_.svm_type = svm_type;
}

/**
 * Set SVM KERNEL to be used
 * 
 * @param[in] svm_kernel_type svm kernel to be used (as defined by libsvm)
 * 
 */
void MriSvm::set_svm_kernel_type(int svm_kernel_type) {
    parameters_.kernel_type = svm_kernel_type;
}

/**
 * Print out all member variables
 */
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

vector<vector<double> > MriSvm::permutated_weights(int number_of_permutations, permutations_array_type &permutations) {
  vector<vector<double> > results;

  struct svm_node *data_base = new svm_node[number_of_samples_];

  for (int permutation_loop(0); permutation_loop < number_of_permutations; permutation_loop++) {
    // Set up a SVM problem with original classes but permutated samples
    for (int sample_loop(0); sample_loop < number_of_samples_; sample_loop++) {
      int shuffled_index = permutations[permutation_loop][sample_loop];

      data_base[sample_loop].values = all_data_[shuffled_index].values;
      data_base[sample_loop].dim    = number_of_features_;
    }
    results.push_back(train_weights(data_base));
  }

  delete[] data_base;

  return results;
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
    permutated_validities[band][row][column][permutation_loop] = cross_validate(leaveout,data_base);
  }
  
  delete[] data_base;
  
  return;
}

/**
 * Cross validate internally stored data
 * 
 * @param[in] leaveout how many samples to be used for testing at any given window
 */
float MriSvm::cross_validate(int leaveout) {
  return cross_validate(leaveout,all_data_);
}

/**
 * Cross validate data_base
 * 
 * @param[in] leaveout  How many samples to be used for testing at any given window
 * @param[in] data_base libsvm data to be used for cross validation
 * 
 * @return  cross validity
 */
float MriSvm::cross_validate(int leaveout,struct svm_node *data_base) {
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
  
  
  // Vector containing the predictions, either predicted classes (SVM) or predicted values (SVR)
  double predictions[number_of_samples_];
  
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
      predictions[prediction_index] = svm_predict(trained_model,&(data_base[prediction_index]));
    }
    /* Free all that unnecessary stuff */
    svm_free_and_destroy_model(&trained_model);
  }
  delete[] trainings_problem.x;
  
  // Calculate score
  
  if(parameters_.svm_type == EPSILON_SVR || parameters_.svm_type == NU_SVR) {
    double total_error = 0.0;
    
    for(int sample_index(0);sample_index<number_of_samples_;sample_index++) {
      double y = classes_[sample_index];
      double v = predictions[sample_index];
      total_error += (v-y)*(v-y);
    }
    return(total_error/number_of_samples_);
  } else {
    int total_correct = 0;
    for(int sample_index(0);sample_index<number_of_samples_;sample_index++)
      if(fabs(predictions[sample_index]-classes_[sample_index]) < 0.000001)
        ++total_correct;
    return ((float) total_correct / (float) (number_of_samples_));
  }

  
}

vector<double> MriSvm::train_weights() {
  return(train_weights(all_data_));
}

vector<double> MriSvm::train_weights(struct svm_node *data_base) {
  vector<double> weights(number_of_features_);

  svm_set_print_string_function(print_nothing);

  struct svm_problem  problem;
  double              classes[number_of_samples_];

  // Prepare
  problem.x = data_base;
  problem.y = classes;
    
  for(int sample_loop(0); sample_loop < number_of_samples_;sample_loop++) {
    classes[sample_loop]            = classes_[sample_loop];
  }
  problem.l = number_of_samples_;

  // Train
  struct svm_model    *model = svm_train(&problem,&parameters_);

  // Retrieve weights
  //cout << "Calculating Weights" << endl;
  for (int i = 0; i < number_of_features_;i++) {
   double w = 0.0;
   for (int j = 0; j < number_of_samples_;j++) {
     w += model->sv_coef[0][j] * model->SV[j].values[i];
   }
   weights[i] = w;
 }

 //cout << endl;
 //cout << "Finished calculating weights" << endl;

 return weights;
}

/**
 * Returns the default svm parameters
 * 
 * @return default svm parameters
 */

struct svm_parameter MriSvm::get_default_parameters() {
  struct svm_parameter parameters;

  parameters.svm_type     = DEFAULT_MRISVM_SVM_TYPE;
  parameters.kernel_type  = DEFAULT_MRISVM_KERNEL_TYPE;
  parameters.degree       = DEFAULT_MRISVM_DEGREE;
  parameters.gamma        = DEFAULT_MRISVM_GAMMA;
  parameters.coef0        = DEFAULT_MRISVM_COEF0;
  parameters.nu           = DEFAULT_MRISVM_NU;
  parameters.cache_size   = DEFAULT_MRISVM_CACHE_SIZE;
  parameters.C            = DEFAULT_MRISVM_C;
  parameters.eps          = DEFAULT_MRISVM_EPS;
  parameters.p            = DEFAULT_MRISVM_P;
  parameters.shrinking    = DEFAULT_MRISVM_SHRINKING;
  parameters.probability  = DEFAULT_MRISVM_PROBABILITY;
  parameters.nr_weight    = DEFAULT_MRISVM_NR_WEIGHT;
  parameters.weight_label = DEFAULT_MRISVM_WEIGHT_LABEL;
  parameters.weight       = DEFAULT_MRISVM_WEIGHT;

  return(parameters);

}


void MriSvm::set_parameters(svm_parameter parameters)
{
  parameters_ = parameters;
}

