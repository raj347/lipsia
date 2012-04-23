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

#include "SearchLight.h"
#include "MriSvm.h"

using std::ofstream;
using std::cerr;
using std::cout;
using std::endl;

SearchLight::SearchLight(int number_of_bands, 
                         int number_of_rows, 
                         int number_of_columns, 
                         int number_of_samples, 
                         sample_3d_array_type sample_features, 
                         vector<int> classes,
                         int radius
               ) : 
                number_of_bands_(number_of_bands), 
                number_of_rows_(number_of_rows), 
                number_of_columns_(number_of_columns), 
                number_of_samples_(number_of_samples), 
                sample_features_(sample_features),
                classes_(classes),
                radius_(radius)
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
  cout << "radius="             << radius_            << endl;
}

vector<coords_3d> SearchLight::radius_pixels() {

  vector<coords_3d> tmp;

  double distance = radius_ * radius_;

   // Search within box
  for (int rel_band(-radius_);rel_band <= radius_; rel_band++)
    for (int rel_row(-radius_);rel_row <= radius_; rel_row++)
      for (int rel_column(-radius_);rel_column <= radius_; rel_column++) {
        if ((rel_band * rel_band + rel_row * rel_row + rel_column * rel_column) <= distance) {
          coords_3d to_insert = { { rel_band, rel_row, rel_column} };
          tmp.push_back(to_insert);
        }
      }
  return tmp;
}

sample_validity_array_type SearchLight::calculate() {
  cout << "Calculating SearchLight" << endl;
  // Find relative coordinates of pixels within radius
  vector<coords_3d> relative_coords = radius_pixels();
  int number_of_features = relative_coords.size();

  sample_features_array_type sample_features(boost::extents[number_of_samples_][number_of_features]);

  // Walk through all voxels of the image data
  sample_validity_array_type validities(boost::extents[number_of_bands_][number_of_rows_][number_of_columns_]);
  boost::progress_display show_progress(number_of_bands_ * number_of_rows_);
  for(int band(0); band < number_of_bands_; band++) {
    for(int row(0); row < number_of_rows_; row++) {
     ++show_progress;
      for(int column(0); column < number_of_columns_; column++) {
        // Check if this is an actual brain pixel
        bool all_zero=true;
        for (int sample(0);sample < number_of_samples_;sample++) {
          if (sample_features_[sample][band][row][column] != 0.0)
            all_zero = false;
        }
        if (!all_zero) {
          // Put stuff into feature fector, to SVM
          int feature_number = 0;
          BOOST_FOREACH(coords_3d coords,relative_coords) {
            int band_coord    = band + coords[0];
            int row_coord     = row + coords[1];
            int column_coord  = column + coords[2];
  
            if ((band_coord >= 0) && (band_coord < number_of_bands_) && (row_coord >= 0) && (row_coord < number_of_rows_) && (column_coord >= 0) && (column_coord < number_of_columns_)) {
              for(int sample(0);sample < number_of_samples_;sample++) {
                sample_features[sample][feature_number] = sample_features_[sample][band_coord][row_coord][column_coord];
              }
              feature_number++;
            }
          }
          MriSvm mrisvm(number_of_samples_,number_of_features,sample_features,classes_);
          validities[band][row][column] = mrisvm.cross_validate();
        } else {
          validities[band][row][column] = 0.0;
        }
      }
    }
  }
  return validities;
}

