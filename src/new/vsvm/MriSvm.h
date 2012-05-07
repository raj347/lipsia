/**
 * MriSvm.h
 *
 *  Created on: 03.04.2012
 *      Author: Tilo Buschmann
 */

#ifndef MRISVM_H
#define MRISVM_H

#include <cfloat>
#include <vector>
#include <string>

#include "boost/multi_array.hpp"

#define DEFAULT_MRISVM_SCALE_LOWER                -1
#define DEFAULT_MRISVM_SCALE_UPPER                1
#define DEFAULT_MRISVM_CROSS_VALIDATION           5
#define DEFAULT_MRISVM_KERNEL_TYPE                LINEAR
#define DEFAULT_MRISVM_SVM_TYPE                   C_SVC
#define DEFAULT_MRISVM_C                          1.0
#define DEFAULT_MRISVM_GAMMA                      1.0
#define DEFAULT_MRISVM_CROSS_VALIDATION_COUNT     10
#define DEFAULT_MRISVM_CROSS_VALIDATION_LEAVEOUT  2

#define _DENSE_REP
#include "libsvm-dense/svm.h"

using std::vector;


typedef boost::multi_array<double, 2> sample_features_array_type;
typedef sample_features_array_type::index sample_features_array_type_index;

class MriSvm {
public:
  enum SVM_TYPE     { C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR };
  enum KERNEL_TYPE  { LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED };

  MriSvm(long int number_of_samples, long int number_of_features, sample_features_array_type sample_features, vector<int> classes);
  MriSvm(long int number_of_samples, long int number_of_features, sample_features_array_type sample_features, vector<int> classes, int, int);

  virtual ~MriSvm();
  void printConfiguration();
  void scale();
  double cross_validate(int count = DEFAULT_MRISVM_CROSS_VALIDATION_COUNT, int leaveout = DEFAULT_MRISVM_CROSS_VALIDATION_LEAVEOUT);
  void export_table(std::string file_name);
  struct svm_parameter getDefaultParameters();


private:
  long int number_of_samples_;
  long int number_of_features_;
  sample_features_array_type sample_features_;
  vector <int>  classes_;
  struct svm_parameter parameters_;
  
  void shuffle(int *array);

};

#endif // MRISVM_H 
