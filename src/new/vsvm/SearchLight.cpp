/**
 * @file SearchLight.cpp
 * Implementation of searchlight support vector machines
 *
 * @author Tilo Buschmann, 2012
 *
 * "I am become death, destroyer of worlds"
 */

// C++ header
#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>

// C header
#include <stdio.h>

// Boost header
#include <boost/foreach.hpp>
#include <boost/progress.hpp>
#include <boost/math/special_functions/factorials.hpp>

// GSL headers
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_permutation.h>

// Class headers
#include "SearchLight.h"

#ifdef _OPENMP
#include <omp.h>
#endif /*_OPENMP*/

using std::ofstream;
using std::cerr;
using std::cout;
using std::endl;

/**
 * Generates a random int
   
   @param[in]     n the exclusive upper limit

   @return random number in the intervall [0,n-1]
*/
static int rand_int(int n) {
  int limit = RAND_MAX - RAND_MAX % n;
  int rnd;
  
  do {
    rnd = rand();
  } while (rnd >= limit);
  return rnd % n;
}

/**
 * The constructor mostly only assigns its parameters to member variables. It also calculates the number of classes used as a preparation for cross validation.
 *
 * @param number_of_bands number of bands
 * @param number_of_rows number of rows
 * @param number_of_columns number of columns
 * @param number_of_samples number of samples
 * @param number_of_features_per_voxel number of features per voxel
 * @param sample_features The data in a 4d array
 * @param classes vector of classes, one class per sample
 * @param extension_band Extension (i.e. measure) of one voxel along the band axis
 * @param extension_row Extension (i.e. measure) of one voxel along the row axis
 * @param extension_column Extension (i.e. measure) of one voxel along the column axis
 */
SearchLight::SearchLight(int number_of_bands, 
                         int number_of_rows, 
                         int number_of_columns, 
                         int number_of_samples, 
                         int number_of_features_per_voxel,
                         sample_3d_array_type sample_features, 
                         vector<int> classes,
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
                extension_band_(extension_band),
                extension_row_(extension_row),
                extension_column_(extension_column),
                do_show_progress(DEFAULT_SEARCHLIGHT_DO_SHOW_PROGRESS)

{
  // Find number of classes from classes vector, but first let's get a copy
  vector<int> classes_copy(classes);
  std::sort(classes_copy.begin(), classes_copy.end());
  vector<int>::iterator it;
  it = std::unique(classes_copy.begin(), classes_copy.end());
 
  number_of_classes_ = it - classes_copy.begin();
}

SearchLight::~SearchLight() {

}

/**
 * Print the values of all member variables.
 */
void SearchLight::printConfiguration() {
  cerr << "SearchLight configuration" << endl;
  cerr << "number_of_bands="    << number_of_bands_   << endl;
  cerr << "number_of_rows="     << number_of_rows_    << endl;
  cerr << "number_of_columns="  << number_of_columns_ << endl;
  cerr << "number_of_samples="  << number_of_samples_ << endl;
  cerr << "number_of_features_per_voxel="  << number_of_features_per_voxel_ << endl;
  cerr << "number_of_classes_"  << number_of_classes_ << endl;
  cerr << "Classes:"                                  << endl;
  for(int sample_loop(0); sample_loop < number_of_samples_; sample_loop++) {
    cerr << classes_[sample_loop] << "\t";
  }
  cerr << endl;
}

/**
 * Calculate which voxels are within a given radius, relativ to a voxel at position (0,0,0)
 */
vector<coords_3d> SearchLight::radius_pixels(double radius) {
  vector<coords_3d> tmp; // Holds relative coordinates of pixels within the radius

  double distance = radius * radius; // square of radius to save a square root later

  // brute force search within a box around (0,0,0)
  for (int rel_band(-radius);rel_band <= radius; rel_band++)
    for (int rel_row(-radius);rel_row <= radius; rel_row++)
      for (int rel_column(-radius);rel_column <= radius; rel_column++) {
        if ((pow(rel_band * extension_band_,2) + pow(rel_row * extension_row_,2) + pow(rel_column * extension_column_,2)) <= distance) {
          coords_3d to_insert = { { rel_band, rel_row, rel_column} };
          tmp.push_back(to_insert);
        }
      }
  return tmp;
}

/**
 * Tests if for every sample, every features of this voxel has a value of 0.0
 *
 * @param band  band coordinate of voxel
 * @param row row coordinate of voxel
 * @column column coordinate of voxel
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

/**
 * Tests if coordinates are within our coordinate system
 *
 *
 * @param band  band coordinate of voxel
 * @param row row coordinate of voxel
 * @column column coordinate of voxel
 */
bool SearchLight::are_coordinates_valid(int band,int row,int column) {
  return((band >= 0) && 
         (band < number_of_bands_) && 
         (row >= 0) && 
         (row < number_of_rows_) && 
         (column >= 0) && 
         (column < number_of_columns_));

}

/**
 * Extract voxel data given by relative coordinates and center voxel given by coordinates and return it in a MriSVM capable format
 *
 * @param[out] sample_features array that is getting filled with mrisvm capable data
 * @param[in] band  band coordinate of voxel
 * @param[in] row row coordinate of voxel
 * @param[in] column column coordinate of voxel
 * @param[in] relative_coords relative coordinates around the given voxel to extract data from
 */
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

/**
 * Calculates cross validation accuracy of searchlight around voxel at position specified by (band,row,colum)
 *
 * The method is leave_out + sliding window, the default leave_out value is equal to the number of classes
 *
 * @param[in] band  band coordinate of voxel
 * @param[in] row row coordinate of voxel
 * @param[in] column column coordinate of voxel
 * @param[in] relative_coords relative coordinates around the given voxel to extract data from
 *
 * @return cross validity of the support vector machine for the searchlight at this voxel
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
  
  return(mrisvm.cross_validate(number_of_classes_));
}

/**
 * Calculates cross validation accuracy of searchlight around voxel at position specified by (band,row,colum), but for a permutated sample, i.e. the samples get thrown around according to the directions givven in the input data
 *
 * The method is leave_out + sliding window, the default leave_out value is equal to the number of classes
 *
 *
 * @param[out]  permutated_validities array to store all the calculated cross validities for all permutations
 * @param[in] number_of_permutations number of permutations
 * @param[in] array containing all permutations
 * @param[in] band  band coordinate of voxel
 * @param[in] row row coordinate of voxel
 * @param[in] column column coordinate of voxel
 * @param[in] relative_coords relative coordinates around the given voxel to extract data from
 */

void SearchLight::cross_validate_permutations(permutated_validities_type  &permutated_validities,
                                              int                         number_of_permutations,
                                              permutations_array_type     &permutations,
                                              int                         band,
                                              int                         row, 
                                              int                         column,
                                              vector<coords_3d>           &relative_coords) {
  int leaveout = number_of_classes_;
  
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

/**
 * Calculate the cross validities via searchlight support vector machines for the whole brain
 *
 * @param[in] radius radius of the searchlight
 *
 * @return array containing cross validities
 */
sample_validity_array_type SearchLight::calculate(double radius) {
  cerr << "Calculating SearchLight" << endl;
  // Find relative coordinates of pixels within radius
  vector<coords_3d> relative_coords = radius_pixels(radius);
  cerr << "Contains " << relative_coords.size() << " voxels " << endl;

  // The result vector
  sample_validity_array_type validities(boost::extents[number_of_bands_][number_of_rows_][number_of_columns_]);

  boost::progress_display show_progress(number_of_bands_ * number_of_rows_);

  // Walk through all voxels of the image data
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

/**
 * Shuffles an array
 *
 * @param[in,out] array array to be shuffled
 */
void SearchLight::shuffle(int *array) {
  int j,tmp;
  
  for (int sample_index(number_of_samples_ - 1); sample_index > 0; sample_index--) {
    j = rand_int(sample_index + 1);
    tmp = array[j];
    array[j] = array[sample_index];
    array[sample_index] = tmp;
  }
  
}

/** 
 *
 * Determines, if this permutation is a "good" permutation, i.e. it is the
 * lexicographical lowest member of the equivalence class
 *
 * @return true if it is a good permutation, false otherwise
 */

bool SearchLight::good_permutation(permutations_array_type &permutations, int
    position,int leaveout) { int count = number_of_samples_ / leaveout; if
  ((number_of_samples_ % leaveout) != 0) count++;
  
  for(int cross_validation_loop(1); cross_validation_loop < count; cross_validation_loop++) {
    if (permutations[position][cross_validation_loop] < permutations[position][cross_validation_loop-1])
      return false;
  }
  return true;
}

/**
 *
 * Convert a permutation to its lexicographic lowest equivalent
 *
 * @param[in,out] permutation to be converted
 * @param[in] how many samples will be left out during cross-validation
 *
 */
void SearchLight::convert_permutation_base(int *permutation,int leaveout) {
  int count = number_of_samples_ / leaveout;
  
  if ((number_of_samples_ % leaveout) != 0)
    count++;

  // Bubble sort, kind of ;)
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

/** 
 *
 * One of several algorithms to generate permutations. This one generates all
 * possible permutations up to a specified maximum. It is a deterministic
 * algorithm, so we cannot consider  this a random drawing
 *
 * @param[in] max_number_of_permutations  number of permutations requested
 * @param[out]  permutations array containing all the newly generated permutations
 *
 * @return actual number of permutations generated, might be smaller than the input parameter max_number_of_permutations because we cannot generate as much
 *
 */

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
    if (good_permutation(permutations,permutation_loop,number_of_classes_)) {
      ++show_progress;
      permutation_loop++;
    }
  }
  while (gsl_permutation_next(p) == GSL_SUCCESS && permutation_loop < new_number_of_permutations);
  
  gsl_permutation_free(p);
  return new_number_of_permutations;
}

/**
 * Tests if two permutations are lexicographical in the equivalence class,i.e. they would  test exactly the same cross-validity
 *
 * @return true if the two permutations are lexicographical equivalent
 *
 */

bool SearchLight::are_permutations_equal(permutations_array_type permutations, int position, int * new_permutation) {
  for (int sample_loop(0); sample_loop < number_of_samples_; sample_loop++) {
    if (permutations[position][sample_loop] != new_permutation[sample_loop])
      return false;
  }
  return true;
}

/**
 * Test if we already know this permutation
 *
 * @return true if permutation is known, false otherwise
 */
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
        convert_permutation_base(shuffle_index,number_of_classes_);
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

SearchLight::PermutationsReturn  SearchLight::calculate_permutations(permutated_validities_type &permutated_validities,int number_of_permutations,double radius) {
  PermutationsReturn permutation_return;
  
  cerr << "Calculating SearchLight Permutations" << endl;
  // Find relative coordinates of pixels within radius
  vector<coords_3d> relative_coords = radius_pixels(radius);

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

