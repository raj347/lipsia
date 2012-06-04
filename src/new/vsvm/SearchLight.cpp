/**
 * SearchLight.cpp
 *
 *  Created on: 16.04.2012
 *      Author: Tilo Buschmann
 */

#include <iostream>
#include <fstream>
#include <algorithm>

#include <stdio.h>

#include <boost/foreach.hpp>
#include <boost/progress.hpp>
#include <boost/math/special_functions/factorials.hpp>

// GSL
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_permutation.h>

#include "SearchLight.h"

#ifdef _OPENMP
#include <omp.h>
#endif /*_OPENMP*/

using std::ofstream;
using std::cerr;
using std::cout;
using std::endl;

/**************************
 * Some necessary C stuff *
 **************************/

/*
 * Get a random int
 */

static int rand_int(int n) {
  int limit = RAND_MAX - RAND_MAX % n;
  int rnd;
  
  do {
    rnd = rand();
  } while (rnd >= limit);
  return rnd % n;
}

/***************
 * Searchlight *
 ***************/
SearchLight::SearchLight(int number_of_bands, 
                         int number_of_rows, 
                         int number_of_columns, 
                         int number_of_samples, 
                         int number_of_features_per_voxel,
                         sample_3d_array_type sample_features, 
                         vector<int> classes,
                         double radius,
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
                extension_band_(extension_band),
                extension_row_(extension_row),
                extension_column_(extension_column),
                do_show_progress(DEFAULT_SEARCHLIGHT_DO_SHOW_PROGRESS)

{
  printConfiguration();
}

SearchLight::~SearchLight() {

}

void SearchLight::printConfiguration() {
  cerr << "SearchLight configuration" << endl;
  cerr << "number_of_bands="    << number_of_bands_   << endl;
  cerr << "number_of_rows="     << number_of_rows_    << endl;
  cerr << "number_of_columns="  << number_of_columns_ << endl;
  cerr << "number_of_samples="  << number_of_samples_ << endl;
  cerr << "feature_dimension="  << number_of_features_per_voxel_ << endl;
  cerr << "radius="             << radius_            << endl;
  cerr << "Classes:"                                  << endl;
  for(int sample_loop(0); sample_loop < number_of_samples_; sample_loop++) {
    cerr << classes_[sample_loop] << "\t";
  }
  cerr << endl;
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

void SearchLight::cross_validate_permutations(permutated_validities_type  &permutated_validities,
                                              int                         number_of_permutations,
                                              permutations_array_type     &permutations,
                                              int                         band,
                                              int                         row, 
                                              int                         column,
                                              vector<coords_3d>           &relative_coords) {
  int leaveout = 2;
  
  int number_of_features = relative_coords.size() * number_of_features_per_voxel_;
  
  sample_features_array_type sample_features(boost::extents[number_of_samples_][number_of_features]);
  
  int feature_number = prepare_for_mrisvm(sample_features,band,row,column,relative_coords);
  MriSvm mrisvm(
                sample_features,
                classes_,
                number_of_samples_,
                feature_number
               );
  
  mrisvm.Permutate(permutated_validities,number_of_permutations,permutations,leaveout,band,row,column);
  return;
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

void SearchLight::shuffle(int *array) {
  int j,tmp;
  
  for (int sample_index(number_of_samples_ - 1); sample_index > 0; sample_index--) {
    j = rand_int(sample_index + 1);
    tmp = array[j];
    array[j] = array[sample_index];
    array[sample_index] = tmp;
  }
  
}

bool SearchLight::good_permutation(permutations_array_type &permutations, int position,int leaveout) {
  int count = number_of_samples_ / leaveout;
  if ((number_of_samples_ % leaveout) != 0)
    count++;
  
  for(int cross_validation_loop(1); cross_validation_loop < count; cross_validation_loop++) {
    if (permutations[position][cross_validation_loop] < permutations[position][cross_validation_loop-1])
      return false;
  }
  return true;
}

void SearchLight::convert_permutation_base(int *permutation,int leaveout) {
  int count = number_of_samples_ / leaveout;
  
  if ((number_of_samples_ % leaveout) != 0)
    count++;

  // Bubble sort, sort of ;)
  for(int outer(1); outer < count; outer++) {
    for(int inner(1); inner < (count-outer+1); inner++) {
      if (permutation[inner] < permutation[inner-1]) {
        // Change around at every permutation step
        for (int prediction_index(inner);prediction_index < number_of_samples_; prediction_index+=count) {
          // Nice clean swap
          int tmp                                    = permutation[prediction_index];
          permutation[prediction_index]   = permutation[prediction_index-1];
          permutation[prediction_index-1] = tmp;
        }
      }
    }
  }
}


int SearchLight::generate_permutations_minimal(int max_number_of_permutations, permutations_array_type &permutations) {
  int n = number_of_samples_;
  int possible_number_of_permutations = static_cast<int>(boost::math::factorial<double>(n) / (boost::math::factorial<double>(n/2)));
  
  int new_number_of_permutations = std::min(max_number_of_permutations,possible_number_of_permutations);
  
  cerr << "Creatinig the minimal number of permutations: " << new_number_of_permutations << endl;
  permutations.resize(boost::extents[new_number_of_permutations][number_of_samples_]);
  
  gsl_permutation *p = gsl_permutation_alloc (number_of_samples_);
  gsl_permutation_init(p); 
  
  boost::progress_display show_progress(new_number_of_permutations);
  int permutation_loop = 0;
  do {
    // Transfer permutation to new position
    for (int sample_loop(0);sample_loop < number_of_samples_; sample_loop++) {
      permutations[permutation_loop][sample_loop] = gsl_permutation_get(p,sample_loop);
    }
    // If it is good, we use the next one
    if (good_permutation(permutations,permutation_loop,2)) {
      ++show_progress;
      permutation_loop++;
    }
  }
  while (gsl_permutation_next(p) == GSL_SUCCESS && permutation_loop < new_number_of_permutations);
  
  gsl_permutation_free(p);
  return new_number_of_permutations;
}

bool SearchLight::are_permutations_equal(permutations_array_type permutations, int position, int * new_permutation) {
  for (int sample_loop(0); sample_loop < number_of_samples_; sample_loop++) {
    if (permutations[position][sample_loop] != new_permutation[sample_loop])
      return false;
  }
  return true;
}
  
bool SearchLight::is_known_permutation(permutations_array_type permutations, int * new_permutation, int number_of_permutations) {
    for (int permutation_loop(0); permutation_loop < number_of_permutations; permutation_loop++) {
      if (are_permutations_equal(permutations,permutation_loop,new_permutation))
        return true;
    }
    return false;
}

int SearchLight::generate_permutations(int max_number_of_permutations,permutations_array_type &permutations) {
  int n = number_of_samples_;
  int possible_number_of_permutations = static_cast<int>(boost::math::factorial<double>(n) / (boost::math::factorial<double>(n/2)));

  cerr << "Number of possible permutations: " << possible_number_of_permutations << endl;
  // If we want more permutations than possible, we give all possible
  if (max_number_of_permutations >= possible_number_of_permutations) {
    cerr << "Too many permutations requested. Will give all known." << endl;
    return(generate_permutations_minimal(possible_number_of_permutations,permutations));
  } else {
    cerr << "I will need to generate a random subset from all possible permutations." << endl;
    // max_number_of_permutations < possible_number_of_permutations
    permutations.resize(boost::extents[max_number_of_permutations][number_of_samples_]);
    
    if (possible_number_of_permutations <= 10000) {
      cerr << "Let me generate all possible permutations and then choose some of them randomly." << endl;
      // Generate all minimal, then draw randomly from them
      permutations_array_type tmp;
      generate_permutations_minimal(possible_number_of_permutations,tmp);
      cerr << "Created " << possible_number_of_permutations << " permutations to choose from." << endl;
      
      // We permutate "possible number of permutations" but take only the first "max_number_of_permutations"
      gsl_permutation *permutation_shuffle = gsl_permutation_alloc(possible_number_of_permutations); 
      gsl_permutation_init(permutation_shuffle);
      
      gsl_rng_env_setup();
      const gsl_rng_type *T = gsl_rng_default;
      gsl_rng *r            = gsl_rng_alloc(T);
      gsl_ran_shuffle (r, permutation_shuffle->data,possible_number_of_permutations, sizeof(size_t));
      
      for (int permutation_loop(0);permutation_loop < max_number_of_permutations; permutation_loop++) {
        int actual_permutation = gsl_permutation_get(permutation_shuffle,permutation_loop);
        // Copy to actual array
        for (int sample_loop(0); sample_loop < number_of_samples_; sample_loop++) {
          permutations[permutation_loop][sample_loop] = tmp[actual_permutation][sample_loop];
        }
      }
      
      gsl_permutation_free(permutation_shuffle);
      gsl_rng_free(r);
      
      return(max_number_of_permutations);
    } else {
      cerr << "Getting random permutations, converting them, then seeing if I already had them." << endl;
      // Draw permutations randomly until we have enough 
      int *shuffle_index = new int[number_of_samples_];
      
      for (int sample_loop(0); sample_loop < number_of_samples_; sample_loop++) {
        shuffle_index[sample_loop] = sample_loop;
      }
      int permutation_loop = 0;
      
      do {
        shuffle(shuffle_index);
        convert_permutation_base(shuffle_index,2);
        if (!is_known_permutation(permutations,shuffle_index,permutation_loop)) {
          for (int sample_loop(0); sample_loop < number_of_samples_; sample_loop++) {
            permutations[permutation_loop][sample_loop] = shuffle_index[sample_loop];
          }
          permutation_loop++;
        }
        
      } while (permutation_loop < max_number_of_permutations);
      delete[] shuffle_index;
    }
    return(max_number_of_permutations);
  }

}

int SearchLight::generate_permutations_deterministic(int max_number_of_permutations,permutations_array_type &permutations) {
  // Find the right size for the permutations array
  int new_number_of_permutations = std::min(max_number_of_permutations,static_cast<int>(boost::math::factorial<long double>(number_of_samples_)));
  permutations.resize(boost::extents[new_number_of_permutations][number_of_samples_]);
  
  int permutation_loop = 0;
 
  gsl_permutation *p = gsl_permutation_alloc (number_of_samples_);
  gsl_permutation_init(p); 
  boost::progress_display show_progress(new_number_of_permutations);
  do {
    ++show_progress;
    for (int sample_loop(0);sample_loop < number_of_samples_; sample_loop++) {
      permutations[permutation_loop][sample_loop] = gsl_permutation_get(p,sample_loop);
    }
    permutation_loop++;
  }
  while (gsl_permutation_next(p) == GSL_SUCCESS && permutation_loop < max_number_of_permutations);
  
  gsl_permutation_free(p);
  return(permutation_loop);
}

int SearchLight::generate_permutations_random(int max_number_of_permutations,permutations_array_type &permutations) {
  // Resize permutations array
  permutations.resize(boost::extents[max_number_of_permutations][number_of_samples_]);
  
  int *shuffle_index = new int[number_of_samples_];
  
  for (int sample_loop(0); sample_loop < number_of_samples_; sample_loop++) {
    shuffle_index[sample_loop] = sample_loop;
  }
  
  for (int permutation_loop(0); permutation_loop < max_number_of_permutations; permutation_loop++) {
    shuffle(shuffle_index);
    
    // fill it into the permutation array
    for (int sample_loop(0); sample_loop < number_of_samples_; sample_loop++) {
      permutations[permutation_loop][sample_loop] = shuffle_index[sample_loop];
    }
    // Now convert base
  }
  delete[] shuffle_index;
  
  return(max_number_of_permutations);
}

void SearchLight::PrintPermutations(int number_of_permutations,permutations_array_type &permutations) {
  for (int permutation_loop(0); permutation_loop < number_of_permutations; permutation_loop++) {
    for (int sample_loop(0); sample_loop < number_of_samples_; sample_loop++) {
      cerr << permutations[permutation_loop][sample_loop] << "\t";
    }
    cerr << endl;
  }
}

SearchLight::PermutationsReturn  SearchLight::calculate_permutations(permutated_validities_type &permutated_validities,int number_of_permutations) {
  PermutationsReturn permutation_return;
  
  cerr << "Calculating SearchLight Permutations" << endl;
  // Find relative coordinates of pixels within radius
  vector<coords_3d> relative_coords = radius_pixels();

  cerr << "Contains " << relative_coords.size() << " voxels " << endl;

  /*******************************************************************************
   * Generate all permutations at once (in preparation for later paralized work) *
   *******************************************************************************/
  
  number_of_permutations = generate_permutations(number_of_permutations,permutation_return.permutations);
  cerr << "New number of permutations: " << number_of_permutations << endl;
  PrintPermutations(number_of_permutations,permutation_return.permutations);
 
  // Walk through all voxels of the image data
  boost::progress_display show_progress(number_of_bands_ * number_of_rows_);

   
#pragma omp parallel for default(none) shared(permutated_validities,permutation_return,number_of_permutations,show_progress,cerr) firstprivate(relative_coords) schedule(dynamic)
  for(int band = 0; band < number_of_bands_; band++) {
   for(int row(0); row < number_of_rows_; row++) {
#pragma omp critical
     ++show_progress;
      for(int column(0); column < number_of_columns_; column++) {
        // Check if this is an actual brain pixel
        if (!(is_voxel_zero(band,row,column))) {
          // Put stuff into feature fector, to SVM
          cross_validate_permutations(permutated_validities, number_of_permutations, permutation_return.permutations, band ,row,column,relative_coords);
        } else {
          for (int permutation_loop(0); permutation_loop < number_of_permutations; permutation_loop++) {
            permutated_validities[band][row][column][permutation_loop] = 0.0;
          }
          
        }
      }
    }
  }
  permutation_return.number_of_permutations = number_of_permutations;
  return permutation_return;
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

