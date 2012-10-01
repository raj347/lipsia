/**
 * @file SearchLight.h
 *
 *  Created on: 16.04.2012
 * @author Tilo Buschmann
 */

#ifndef SEARCHLIGHT_H
#define SEARCHLIGHT_H

#include <cfloat>
#include <vector>
#include <string>

#define BOOST_DISABLE_ASSERTS
#include "boost/multi_array.hpp"

#define DEFAULT_SEARCHLIGHT_SCALE_LOWER 0
#define DEFAULT_SEARCHLIGHT_SCALE_UPPER 1
#define DEFAULT_SEARCHLIGHT_RADIUS      7
#define DEFAULT_SEARCHLIGHT_DO_SHOW_PROGRESS false
#define DEFAULT_SEARCHLIGHT_LEAVEOUT 2

#include "MriTypes.h"
#include "MriSvm.h"

#define _DENSE_REP
#include "libsvm-dense/svm.h"

using std::vector;

struct vector_lex_compare {
  bool operator() (const vector<int>& x, const vector<int>& y) const{
    return std::lexicographical_compare(x.begin(), x.end(), y.begin(), y.end());
  }
};


class SearchLight {
public:
  // Methods
  SearchLight(int number_of_bands,
              int number_of_rows,
              int number_of_columns,
              int number_of_samples,
              int number_of_features_per_voxel,
              sample_3d_array_type sample_features,
              vector<double> classes,
              double extension_band,
              double extension_row,
              double extension_column
      );

  virtual ~SearchLight();
  void printConfiguration();
  static int generate_permutations(int, int, int max_number_of_permutations,permutations_array_type &permutations);

  sample_validity_array_type calculate(double radius);
  int calculate_permutations(permutated_validities_type &permutated_validities, permutations_array_type &permutations, int number_of_permutations, double radius);
  void scale();
  void set_parameters(struct svm_parameter);
  
  

private:
  static void shuffle(vector <int> &shuffle,int n);
  static int generate_permutations_minimal(int number_of_samples, int number_of_classes, int max_number_of_permutations, permutations_array_type &permutations);
  static bool is_known_permutation(int number_of_samples, permutations_array_type permutations, int * new_permutation, int number_of_permutations);
  static bool are_permutations_equal(int number_of_samples, permutations_array_type permutations, int position, int * new_permutation);
  
  static bool good_permutation(int number_of_samples, permutations_array_type &permutations, int position,int leaveout);
  static void convert_permutation_base(int number_of_samples, vector<int> &permutation,int leaveout);
  
  void PrintPermutations(int number_of_permutations,permutations_array_type &permutations);
  
  vector<coords_3d>     radius_pixels(double);
  bool                  is_voxel_zero(int,int,int);
  bool                  are_coordinates_valid(int,int,int);
  double                cross_validate(int,int,int,vector<coords_3d>&);
  struct svm_parameter  getDefaultParameters();
  void                  scale_voxel_feature(int band, int row, int column,int feature);
  int                   prepare_for_mrisvm(sample_features_array_type &sample_features,int band, int row, int column,vector<coords_3d> &relative_coords);
  void                  cross_validate_permutations(permutated_validities_type &permutated_validities,
                                                    int number_of_permutations,
                                                    permutations_array_type &permutations,
                                                    int band, 
                                                    int row, 
                                                    int column,
                                                    vector<coords_3d> &relative_coords);

  int      number_of_bands_;
  int      number_of_rows_;
  int      number_of_columns_;
  int      number_of_samples_;
  int      number_of_features_per_voxel_;
  int      number_of_classes_;
  
  sample_3d_array_type sample_features_;
  vector <double>  classes_;

  int     svm_type_;
  int     svm_kernel_type_;
  
  double  extension_band_,extension_row_,extension_column_;
  
  bool    do_show_progress;
  
  struct svm_parameter parameters_;

};

#endif // SEARCHLIGHT_H 
