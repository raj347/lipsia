/**
 * @file Threshold.cpp
 *
 * @author Tilo Buschmann
 * @date 03.04.2012
 * 
 */

#include <iostream>
#include <fstream>
#include <algorithm>

// GSL headers
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>

#include <boost/progress.hpp>

#include "Threshold.h"

using std::ofstream;
using std::cerr;
using std::cout;
using std::endl;

Threshold::Threshold() 
{
  //printConfiguration();
}

/**
 */
Threshold::~Threshold() {
}

boost::multi_array<float, 4> Threshold::calculate(boost::multi_array<float, 4> &pool, float p, bool is_two_sided) {
  int number_of_bands         = pool.shape()[0];
  int number_of_rows          = pool.shape()[1];
  int number_of_columns       = pool.shape()[2];
  int pool_size               = pool.shape()[3];

  cerr << "Dimensions: " << number_of_bands << "/" << number_of_rows << "/" << number_of_columns << "/" << pool_size << endl;
  boost::multi_array<float, 4> thresholds;

  if (is_two_sided)
    thresholds.resize(boost::extents[number_of_bands][number_of_rows][number_of_columns][2]);
  else
    thresholds.resize(boost::extents[number_of_bands][number_of_rows][number_of_columns][1]);

  boost::progress_display histo_progress(number_of_bands * number_of_rows * number_of_columns);
  cerr << "Calculating individual voxel thresholds." << endl;

#pragma omp parallel for default(none) shared(number_of_bands, number_of_rows, number_of_columns, pool_size, histo_progress, thresholds, pool, is_two_sided, p) schedule(dynamic)
  for(int band = 0; band < number_of_bands; band++) {
    for(int row(0); row < number_of_rows; row++) {
      for(int column(0); column < number_of_columns; column++) {
        vector<float> voxel_pool(pool_size);
       
        for (int pool_index(0); pool_index < pool_size; pool_index++) {
          //cerr << "Accessing: " << band << "/" << row << "/" << column << "/" << pool_index << endl;
          voxel_pool[pool_index] = pool[band][row][column][pool_index];
        }

        std::sort(voxel_pool.begin(), voxel_pool.end());

        if (is_two_sided) {
          int threshold_position_left   = std::max((int) floor( (p/2.0) * pool_size),0);
          int threshold_position_right  = std::min((int) ceil((1.0 - p/2.0) * pool_size) - 1, pool_size);

#pragma omp critical 
          {
            thresholds[band][row][column][0] = voxel_pool[threshold_position_right];
            thresholds[band][row][column][1] = voxel_pool[threshold_position_left];
          }
        } else {
          int threshold_position = std::min((int) ceil( (1.0 - p) * pool_size) - 1, pool_size);
#pragma omp critical 
          thresholds[band][row][column][0] = voxel_pool[threshold_position];
        }
#pragma omp critical 
        ++histo_progress;
      }
    }
  }
  cerr << "Done." << endl;
  return(thresholds);
}

