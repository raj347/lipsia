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

#define DEFAULT_SEARCHLIGHT_SCALE_LOWER -1
#define DEFAULT_SEARCHLIGHT_SCALE_UPPER 1

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
      int radius);

  virtual ~SearchLight();
  void printConfiguration();
  sample_validity_array_type calculate();

private:
  vector<coords_3d> radius_pixels();

  int      number_of_bands_;
  int      number_of_rows_;
  int      number_of_columns_;
  int      number_of_samples_;
  sample_3d_array_type sample_features_;
  vector <int>  classes_;
  int           radius_;

};

#endif // SEARCHLIGHT_H 
