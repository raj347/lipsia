/**
 * @file MriSvm.h
 *  
 * @author Tilo Buschmann
 * @date  03.04.2012
 */

#ifndef MRISVM_H
#define MRISVM_H

#include <cfloat>
#include <vector>
#include <string>

//#define BOOST_DISABLE_ASSERTS
#include "boost/multi_array.hpp"

#define DEFAULT_MRISVM_SCALE_LOWER                -1
#define DEFAULT_MRISVM_SCALE_UPPER                1

#define DEFAULT_MRISVM_KERNEL_TYPE  LINEAR
#define DEFAULT_MRISVM_SVM_TYPE     C_SVC
#define DEFAULT_MRISVM_DEGREE       3
#define DEFAULT_MRISVM_GAMMA        1.0
#define DEFAULT_MRISVM_COEF0        0.0
#define DEFAULT_MRISVM_NU           0.5
#define DEFAULT_MRISVM_CACHE_SIZE   100
#define DEFAULT_MRISVM_C            1
#define DEFAULT_MRISVM_EPS          0.1
#define DEFAULT_MRISVM_P            0.1
#define DEFAULT_MRISVM_SHRINKING    1
#define DEFAULT_MRISVM_PROBABILITY  0
#define DEFAULT_MRISVM_NR_WEIGHT    0
#define DEFAULT_MRISVM_WEIGHT_LABEL NULL
#define DEFAULT_MRISVM_WEIGHT       NULL

#define _DENSE_REP
#include "libsvm-dense/svm.h"

#include "MriTypes.h"
#include "SearchLight.h"

using std::vector;

class MriSvm {
public:
  enum SVM_TYPE     { C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR };
  enum KERNEL_TYPE  { LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED };

  MriSvm(
         sample_features_array_type sample_features, 
         vector<double>             classes,
         long int                   number_of_samples, 
         long int                   number_of_features
        );

  virtual ~MriSvm();
  
  void    printConfiguration();
  void    scale();
  float   cross_validate(int leaveout);
  float   cross_validate(int leaveout,struct svm_node *all_data);

  void    train_weights(boost::multi_array<double,1> &weights);
  void    train_weights(boost::multi_array<double,1> &weights, struct svm_node *data_base);

  void permutated_weights(boost::multi_array<double, 2> &weight_matrix, int number_of_permutations, permutations_array_type &permutations);
  void    Permutate(permutated_validities_type &permutated_validities,
                       int number_of_permutations,
                       permutations_array_type &permutations,
                       int leaveout,
                       int band, 
                       int row, 
                       int column); // FIXME MriSVM should not be aware of coordinates

  void    set_svm_type(int svm_type);
  void    set_svm_kernel_type(int svm_kernel_type);
  static  struct svm_parameter get_default_parameters();
  void set_parameters(svm_parameter);

  static int generate_permutations(int n, int max_number_of_permutations, permutations_array_type &permutations);

private:
  void construct(sample_features_array_type  &sample_features);
  void prepare_all_data(sample_features_array_type  &sample_features);
  
  //sample_features_array_type  sample_features_;
  vector<double>        classes_;
  long int              number_of_samples_;
  long int              number_of_features_;
  struct svm_parameter  parameters_;
  struct svm_node       *all_data_;
  

};

#endif // MRISVM_H 
