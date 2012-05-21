/**
 * SearchLight.cpp
 *
 *  Created on: 16.04.2012
 *      Author: Tilo Buschmann
 */

#include <iostream>
#include <fstream>

#include <boost/foreach.hpp>
#include <boost/progress.hpp>
#include <boost/math/special_functions/factorials.hpp>

#include "SearchLight.h"

#ifdef _OPENMP
#include <omp.h>
#endif /*_OPENMP*/

using std::ofstream;
using std::cerr;
using std::cout;
using std::endl;

SearchLight::SearchLight(int number_of_bands, 
                         int number_of_rows, 
                         int number_of_columns, 
                         int number_of_samples, 
                         int number_of_features_per_voxel,
                         sample_3d_array_type sample_features, 
                         vector<int> classes,
                         double radius,
                         int svm_type,
                         int svm_kernel_type,
                         double extension_band,
                         double extension_row,
                         double extension_column
               ) : 
                number_of_bands_(number_of_bands), 
                number_of_rows_(number_of_rows), 
                number_of_columns_(number_of_columns), 
                number_of_samples_(number_of_samples), 
                number_of_features_per_voxel_(number_of_features_per_voxel),
                sample_features_(sample_features),
                classes_(classes),
                radius_(radius),
                svm_type_(svm_type),
                svm_kernel_type_(svm_kernel_type),
                extension_band_(extension_band),
                extension_row_(extension_row),
                extension_column_(extension_column),
                do_show_progress(DEFAULT_SEARCHLIGHT_DO_SHOW_PROGRESS)

{
  //printConfiguration();
}

SearchLight::~SearchLight() {

}

void SearchLight::printConfiguration() {
  cout << "SearchLight configuration" << endl;
  cout << "number_of_bands="    << number_of_bands_   << endl;
  cout << "number_of_rows="     << number_of_rows_    << endl;
  cout << "number_of_columns="  << number_of_columns_ << endl;
  cout << "number_of_samples="  << number_of_samples_ << endl;
  cout << "feature_dimension="  << number_of_features_per_voxel_ << endl;
  cout << "radius="             << radius_            << endl;
}

/*
 * Calculates all relative pixels within a given radius
 */
vector<coords_3d> SearchLight::radius_pixels() {
  vector<coords_3d> tmp; // Holds relative coordinates of pixels within the radius

  double distance = radius_ * radius_; // square of radius to safe a square root later

   // Search within box
  for (int rel_band(-radius_);rel_band <= radius_; rel_band++)
    for (int rel_row(-radius_);rel_row <= radius_; rel_row++)
      for (int rel_column(-radius_);rel_column <= radius_; rel_column++) {
        if ((pow(rel_band * extension_band_,2) + pow(rel_row * extension_row_,2) + pow(rel_column * extension_column_,2)) <= distance) {
          coords_3d to_insert = { { rel_band, rel_row, rel_column} };
          tmp.push_back(to_insert);
        }
      }
  return tmp;
}

/*
 * Tests if feature has a value of 0.0 in every sample
 */
bool SearchLight::is_voxel_zero(int band,int row,int column) {
  for (int sample(0);sample < number_of_samples_;sample++) {
    for(int feature_index(0); feature_index < number_of_features_per_voxel_; feature_index++) {
      if (sample_features_[sample][band][row][column][feature_index] != 0.0)
        return(false);
    }
  }
  return(true);
}

/*
 * Tests if coordinates are within our coordinate system
 */
bool SearchLight::are_coordinates_valid(int band,int row,int column) {
  return((band >= 0) && 
         (band < number_of_bands_) && 
         (row >= 0) && 
         (row < number_of_rows_) && 
         (column >= 0) && 
         (column < number_of_columns_));

}

int SearchLight::prepare_for_mrisvm(sample_features_array_type &sample_features,int band, int row, int column,vector<coords_3d> &relative_coords) {

  int feature_number = 0;
  BOOST_FOREACH(coords_3d coords,relative_coords) {
    int band_coord    = band    + coords[0];
    int row_coord     = row     + coords[1];
    int column_coord  = column  + coords[2];

    if (are_coordinates_valid(band_coord,row_coord,column_coord)) {
      for(int feature_index(0); feature_index < number_of_features_per_voxel_; feature_index++) {
        for(int sample(0);sample < number_of_samples_;sample++) {
          sample_features[sample][feature_number] = sample_features_[sample][band_coord][row_coord][column_coord][feature_index];
        }
        feature_number++;
      }
    }
  }
  return feature_number;
}

/*
 * Calculates cross validation accuracy of voxel at position specified by (band,row,colum)
 */
double SearchLight::cross_validate(int band, int row, int column,vector<coords_3d> &relative_coords) {
  int number_of_features = relative_coords.size() * number_of_features_per_voxel_;
  
  sample_features_array_type sample_features(boost::extents[number_of_samples_][number_of_features]);
  
  int feature_number = prepare_for_mrisvm(sample_features,band,row,column,relative_coords);
  MriSvm mrisvm(
                sample_features,
                classes_,
                number_of_samples_,
                feature_number
               );
  
  return(mrisvm.cross_validate(2));
}

/*
 * Calculates cross validation accuracy for all voxels
 */
sample_validity_array_type SearchLight::calculate() {
  cerr << "Calculating SearchLight" << endl;
  // Find relative coordinates of pixels within radius
  vector<coords_3d> relative_coords = radius_pixels();

  cerr << "Contains " << relative_coords.size() << " voxels " << endl;

  sample_validity_array_type validities(boost::extents[number_of_bands_][number_of_rows_][number_of_columns_]);

  // Walk through all voxels of the image data
  boost::progress_display show_progress(number_of_bands_ * number_of_rows_);

#pragma omp parallel for default(none) shared(validities,show_progress) firstprivate(relative_coords) schedule(dynamic)
  for(int band = 0; band < number_of_bands_; band++) {
    for(int row(0); row < number_of_rows_; row++) {
#pragma omp critical
     ++show_progress;
      for(int column(0); column < number_of_columns_; column++) {
        // Check if this is an actual brain pixel
        if (!(is_voxel_zero(band,row,column))) {
          // Put stuff into feature fector, to SVM
          validities[band][row][column] = cross_validate(band,row,column,relative_coords);
        } else {
          validities[band][row][column] = 0.0;
        }
      }
    }
  }
  return validities;
}

vector<sample_validity_array_type> SearchLight::calculate_permutations(int number_of_permutations) {
  cerr << "Calculating SearchLight Permutations" << endl;
  // Find relative coordinates of pixels within radius
  vector<coords_3d> relative_coords = radius_pixels();

  cerr << "Contains " << relative_coords.size() << " voxels " << endl;

  sample_validity_array_type validity(boost::extents[number_of_bands_][number_of_rows_][number_of_columns_]);

  // Walk through all voxels of the image data
  boost::progress_display show_progress(number_of_bands_ * number_of_rows_);

#pragma omp parallel for default(none) shared(validity,show_progress) firstprivate(relative_coords) schedule(dynamic)
  for(int band = 0; band < number_of_bands_; band++) {
    for(int row(0); row < number_of_rows_; row++) {
#pragma omp critical
     ++show_progress;
      for(int column(0); column < number_of_columns_; column++) {
        // Check if this is an actual brain pixel
        if (!(is_voxel_zero(band,row,column))) {
          // Put stuff into feature fector, to SVM
          validity[band][row][column] = cross_validate(band,row,column,relative_coords);
        } else {
          validity[band][row][column] = 0.0;
        }
      }
    }
  }
  vector<sample_validity_array_type> validities;
  validities.push_back(validity);
  return validities;
}

/*
 * Scales all features
 */

void SearchLight::scale_voxel_feature(int band, int row, int column,int feature) {
  // Find range of values of this feature
  double max = DBL_MIN;
  double min = DBL_MAX;
  for (long int sample_index(0); sample_index < number_of_samples_; sample_index++) {
    double x = sample_features_[sample_index][band][row][column][feature];
    if (x < min) {
      min = x;
    }
    if (x > max) {
      max = x;
    }
  }
  
  // Scale
  for (long int sample_index(0); sample_index < number_of_samples_; sample_index++) {
    double x = sample_features_[sample_index][band][row][column][feature];
    double lower = DEFAULT_SEARCHLIGHT_SCALE_LOWER;
    double upper = DEFAULT_SEARCHLIGHT_SCALE_UPPER;
    
    sample_features_[sample_index][band][row][column][feature] = lower + (upper - lower) * (x - min) / (max - min);
  }
}

void SearchLight::scale() {
  cout << "Scaling" << endl;
  boost::progress_display show_progress(number_of_bands_ * number_of_rows_);
  for(int band(0); band < number_of_bands_; band++) {
    for(int row(0); row < number_of_rows_; row++) {
      ++show_progress;
      for(int column(0); column < number_of_columns_; column++) {
        if (!(is_voxel_zero(band,row,column)))
          for(int feature(0); feature < number_of_features_per_voxel_; feature++) {
            scale_voxel_feature(band,row,column,feature);
          }
      }
    }
  }
}

