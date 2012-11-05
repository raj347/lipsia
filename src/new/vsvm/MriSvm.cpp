/**
 * @file MriSvm.cpp
 *
 * @author Tilo Buschmann
 * @date 03.04.2012
 * 
 */

#include <iostream>
#include <fstream>

#include <boost/foreach.hpp>
#include <boost/progress.hpp>

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
  parameters_       = get_default_parameters();
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
      //cerr << sample_features[sample_index][feature_index] << " ";
    }
    //cerr << endl;
  }
}

/**
 * Rescale data to be in the distribution N(0,1)
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
 * Print out all member variables (used for debugging)
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

/**
 * Calculate SVM weights for a number of permutations
 *
 * @param[in] number_of_permutations number of permutations
 * @param[in] permutations array containing all permutations
 * 
 * @return one vector of weights per permutation
 */

void MriSvm::permutated_weights(boost::multi_array<double, 2> &weight_matrix, int number_of_permutations, permutations_array_type &permutations) {

  boost::progress_display show_progress(number_of_permutations);

#pragma omp parallel for default(none) shared(number_of_permutations, show_progress, permutations, weight_matrix) private(boost::extents) schedule(dynamic)
  for (int permutation_loop = 0; permutation_loop < number_of_permutations; permutation_loop++) {
    struct svm_node *data_base = new svm_node[number_of_samples_];
    boost::multi_array<double, 1> weights(boost::extents[number_of_features_]);
    // Set up a SVM problem with original classes but permutated samples
    for (int sample_loop(0); sample_loop < number_of_samples_; sample_loop++) {
      int shuffled_index = permutations[permutation_loop][sample_loop];

      data_base[sample_loop].values = all_data_[shuffled_index].values;
      data_base[sample_loop].dim    = number_of_features_;
    }
    train_weights(weights,data_base);
    for (int feature_index(0); feature_index < number_of_features_; feature_index++) {
      weight_matrix[permutation_loop][feature_index] = weights[feature_index];
    }
    delete [] data_base;

#pragma omp critical
    ++show_progress;
  }

  return;
}

/**
 * Calculate cross validities for this problem for all given permutations
 *
 * @param[out]  permutated_validities cross validation values for all permutations
 * @param[in] number_of_permutations number of permutations
 * @param[in] permutations array containing all permutations
 * @param[in] leaveout leaveout value for cross validation (i.e. @leaveout samples will be used for testing, the rest for training, sliding window style)
 * @param[in] band band coordinate of voxel
 * @param[in] row row coordinate of voxel
 * @param[in] column  column coordinate of voxel
 * 
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

/**
 *  Train weights on all data
 *
 *  @return SVM weights
 */
void MriSvm::train_weights(boost::multi_array<double,1> &weights) {
  train_weights(weights,all_data_);
  return;
}

/**
 * Train weights on @data_base
 *
 * @param[in] data_base data base to train on
 *
 * @return SVM weights
 */
void MriSvm::train_weights(boost::multi_array<double,1> &weights, struct svm_node *data_base) {
  svm_set_print_string_function(print_nothing);

  struct svm_problem  problem;
  double              classes[number_of_samples_];

  // Prepare
  problem.x = data_base;
  problem.y = classes;
    
  for(int sample_loop(0); sample_loop < number_of_samples_;sample_loop++) {
    classes[sample_loop]            = classes_[sample_loop];
   //cerr << " " << classes[sample_loop];
  }
  //cerr << endl;
  problem.l = number_of_samples_;

  // Train
  struct svm_model    *model = svm_train(&problem,&parameters_);

  // Retrieve weights
  for (int i = 0; i < number_of_features_;i++) {
   double w = 0.0;
   for (int j = 0; j < model->l;j++) {
     w += model->sv_coef[0][j] * model->SV[j].values[i];
   }
   weights[i] = w;
   //cerr << " " << w;
  }
  //cerr << endl;

 return;
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

/**
 * Set SVM parameters
 *
 * @param[in] parameters svm parameters (libsvm format)
 */
void MriSvm::set_parameters(svm_parameter parameters)
{
  parameters_ = parameters;
}

/*
int MriSvm::generate_permutations(int n, int max_number_of_permutations, permutations_array_type &permutations) {
      permutations.resize(boost::extents[2][n]);
      int permutation_0[40] = {37,20,17,5,35,16,13,11,6,19,28,8,1,22,14,26,15,0,12,3,24,29,9,34,32,2,33,23,31,38,39,10,21,27,36,18,4,7,25,30};
      int permutation_1[40] = {34,13,2,32,16,14,25,33,38,18,21,29,17,22,8,9,23,11,39,10,35,24,20,19,27,15,31,3,30,5,4,28,6,0,37,7,12,1,36,26};

      for (int sample_loop(0); sample_loop < n; sample_loop++) {
        permutations[0][sample_loop] = permutation_0[sample_loop];
        permutations[1][sample_loop] = permutation_1[sample_loop];
      }

      return 2;
}
*/

int MriSvm::generate_permutations(int n, int max_number_of_permutations, permutations_array_type &permutations) {
      permutations.resize(boost::extents[max_number_of_permutations][n]);
      vector<int> shuffle_index(n);

      for (int sample_loop(0); sample_loop < n; sample_loop++) {
        shuffle_index[sample_loop] = sample_loop;
      }

      for (int permutation_loop(0); permutation_loop < max_number_of_permutations; permutation_loop++) {
        //cerr << "Permutation number " << permutation_loop << endl;
        SearchLight::shuffle(shuffle_index,n);
        for (int sample_loop(0); sample_loop < n; sample_loop++) {
          permutations[permutation_loop][sample_loop] = shuffle_index[sample_loop];
        }
      }
      return max_number_of_permutations;

}

