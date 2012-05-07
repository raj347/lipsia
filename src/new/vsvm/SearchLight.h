/**
 * SearchLight.h
 *
 *  Created on: 16.04.2012
 *      Author: Tilo Buschmann
 */

#ifndef SEARCHLIGHT_H
#define SEARCHLIGHT_H

#include <cfloat>
#include <vector>
#include <string>

#include "boost/multi_array.hpp"

#define DEFAULT_SEARCHLIGHT_SCALE_LOWER 0
#define DEFAULT_SEARCHLIGHT_SCALE_UPPER 1
#define DEFAULT_SEARCHLIGHT_RADIUS      3

#include "MriSvm.h"

#define _DENSE_REP
#include "libsvm-dense/svm.h"

using std::vector;

typedef boost::multi_array<double, 4> sample_3d_array_type;
typedef boost::multi_array<double, 3> sample_validity_array_type;
typedef boost::array<int,3>           coords_3d;

class SearchLight {
public:
  SearchLight(int number_of_bands, 
      int number_of_rows, 
      int number_of_columns, 
      int number_of_samples, 
      sample_3d_array_type sample_features, 
      vector<int> classes,
      double radius,
      int svm_type,
      int svm_kernel_type,
      double extension_band,
      double extension_row,
      double extension_column
      );

  virtual ~SearchLight();
  void printConfiguration();
  sample_validity_array_type calculate();
  void scale();

private:
  vector<coords_3d> radius_pixels();
  bool              is_feature_zero(int,int,int);
  bool              are_coordinates_valid(int,int,int);
  double            cross_validate(int,int,int,vector<coords_3d>&);
  struct svm_parameter getDefaultParameters();

  int      number_of_bands_;
  int      number_of_rows_;
  int      number_of_columns_;
  int      number_of_samples_;
  sample_3d_array_type sample_features_;
  vector <int>  classes_;
  double  radius_;

  int     svm_type_;
  int     svm_kernel_type_;
  
  double extension_band_,extension_row_,extension_column_;

};

#endif // SEARCHLIGHT_H 
