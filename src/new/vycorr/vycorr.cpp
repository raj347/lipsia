/**
 * @file vycorr.cpp
 * 
 * Correlates every voxel independently to another separate variable
 *
 * Usage:
 *  vycorr -in samples.v -corr correlations.txt [-j nprocs]
 *
 *  options:
 *    -in samples.v
 *      sample vista files files
 *    -corr correlations.txt
 *      name of files containing the correlations, one per line for every sample
 *    -j nprocs
 *      number of processors to use, '0' to use all
 *
 * @author Tilo Buschmann
 * @date 24.8.2012
 *
 */

// C++ header
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <map>

// C header
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Boost header
#define BOOST_DISABLE_ASSERTS // increases performance, disable when debugging
#include <boost/multi_array.hpp>
#include <boost/assign.hpp>
#include <boost/progress.hpp>
#include <boost/foreach.hpp>

// VIA header
#include <viaio/Vlib.h>
#include <viaio/VImage.h>
#include <viaio/mu.h>
#include <viaio/option.h>

#include "compat.h"

// Some stuff I am using from boost and std lib, explicitly declared
using std::cerr;
using std::cout;
using std::endl;
using std::setw;
using std::vector;
using std::map;
using std::string;
using std::ofstream;
using std::ifstream;
using boost::assign::map_list_of;

extern "C" void getLipsiaVersion(char*,size_t);

#ifdef _OPENMP
#include <omp.h>
#endif /*_OPENMP*/

#ifdef _OPENMP
/**
 * Configure OpenMP
 * 
 * @param[in] nproc number of processing cores to be used
 */
void configure_omp(int nproc) {
  int number_of_cores = omp_get_num_procs();
  if (nproc > 0 && nproc < number_of_cores) 
    number_of_cores = nproc;
  printf("Using %d cores\n",number_of_cores);
  omp_set_num_threads(number_of_cores);
}
#endif /*OPENMP */

double artanh(double x) {
  return(0.5 * log((1+x)/(1-x)));
}

/**
 * Format of input:
 * - One VIA file per subject/sample
 * - One image within VIA file per feature
 * - Same dimensions and attributes for every image, same number of features per sample
 */
int main (int argc,char *argv[]) {
  /********************
   * Initialise Vista *
   ********************/

  // Output program name and version
  char version[100];
  getLipsiaVersion(version, sizeof(version));
  cerr << argv[0] << " V" << version << endl;

  // Names of input files containing VIA data
  VShort            nproc           = 4;
  static VArgVector input_filenames;
  VString correlation_filename = NULL;

  FILE *out_file;

  static VOptionDescRec program_options[] = {
    {"in",  VStringRepn, 0, &input_filenames,       VRequiredOpt, NULL, "Input files" },
    {"corr",VStringRepn, 1, &correlation_filename,  VRequiredOpt, NULL, "File containing the values to be correlated" },
    {"j",   VShortRepn,  1, &nproc,                 VOptionalOpt, NULL, "number of processors to use, '0' to use all" }
  };
  VParseFilterCmd(VNumber(program_options),program_options,argc,argv,NULL,&out_file);

#ifdef _OPENMP
  // Take care of multiprocessing
  configure_omp(nproc); 
#endif /*OPENMP */
  
  int number_of_samples             = input_filenames.number;
  
  /*************************
   * Read Correlation File *
   *************************/
  double y[number_of_samples];

  //cerr << "Opening file" << correlation_filename << endl;
  ifstream input(correlation_filename);

  double y_mean = 0.0;
  double f;
  int sample_index;
  for (sample_index = 0, f = 0.0; (input >> f) && (sample_index < number_of_samples); sample_index++) {
    //cerr << "Read value " << f << " at sample_index " << sample_index << endl;
    y[sample_index] = f;
    y_mean += f;
  }
  y_mean /= number_of_samples;
          
  double y_sd = 0.0;
  for (int sample_index(0); sample_index < number_of_samples;sample_index++) {
    double diff = y[sample_index] - y_mean;
    y_sd  += diff * diff;
  }
  y_sd = sqrt(y_sd/((double) (number_of_samples-1)));

  cerr << "y_mean=" << y_mean << " y_sd=" << y_sd << endl;

  /*************************************
   * Read image files and extract data *
   *************************************/


  int number_of_voxels              = 0;
  int number_of_features_per_voxel  = 0;

  vector<VImage> *source_images = new vector<VImage>[number_of_samples];

  VAttrList attribute_list;
  
  for (int sample_index(0); sample_index < number_of_samples; sample_index++) {
      /*******************
       * Read VIA file *
       *******************/
      VStringConst input_filename = ((VStringConst *) input_filenames.vector)[sample_index];
      cerr << input_filename << " (y=" << y[sample_index] << ")" << endl;
      FILE *input_file            = VOpenInputFile(input_filename, TRUE);
      attribute_list              = VReadFile(input_file, NULL);
      fclose(input_file);

      
      if(!attribute_list)
        VError("Error reading image");
    
      /**********************
       * Analyse attributes *
       **********************/

      VAttrListPosn position;
      int this_number_of_features_per_voxel = 0;

      // Walk through all images in this file
      for (VFirstAttr(attribute_list, &position); VAttrExists(&position); VNextAttr(&position)) {
        // Skip attribute if it is not an image
        if (VGetAttrRepn(&position) != VImageRepn)
          continue;

        // Extract image
        VImage image;
        VGetAttrValue(&position,NULL,VImageRepn,&image);

        // Put it into the image vector of this sample and increase number of features per voxel for this sample
        source_images[sample_index].push_back(image);
        this_number_of_features_per_voxel++;

        /*************************
         * Consistency check (1) *
         *************************/
        // Get number of voxels (i.e. bands * rows * columns)
        int this_number_of_voxels = VImageNPixels(image);
        if (0 == number_of_voxels) {
          number_of_voxels = this_number_of_voxels;
        } else if (number_of_voxels != this_number_of_voxels) {
          VError("Error: Number of features differs from number of features in previous pictures.");
        }
      }
      /***************************
       * Consistency check (2+3) *
       ***************************/
      if (this_number_of_features_per_voxel == 0)
        VError("No input image found");

      if (number_of_features_per_voxel == 0) {
        number_of_features_per_voxel = this_number_of_features_per_voxel;
      } else if (number_of_features_per_voxel != this_number_of_features_per_voxel) {
        VError("This file has a different number of images than previous files.");
      }
  }

  int number_of_bands   = VImageNBands(source_images[0].front());
  int number_of_rows    = VImageNRows(source_images[0].front());
  int number_of_columns = VImageNColumns(source_images[0].front());

  /* Measure time */
  struct timespec start,end;
  lipsia_gettime(&start);

  VAttrList out_list = VCreateAttrList();
  for(int feature_index(0); feature_index < number_of_features_per_voxel; feature_index++) {
    VImage dest = VCreateImage(number_of_bands,number_of_rows,number_of_columns,VFloatRepn);
    VImage z_dest = VCreateImage(number_of_bands,number_of_rows,number_of_columns,VFloatRepn);
    VFillImage(dest,VAllBands,0);
    VFillImage(z_dest,VAllBands,0);
    VCopyImageAttrs (source_images[0].front(), dest);
    VCopyImageAttrs (source_images[0].front(), z_dest);
   
    for (int band(0); band < number_of_bands;band++) {
      for (int row(0); row < number_of_rows;row++) {
        for (int column(0); column < number_of_columns;column++) {
          // Caculate voxel mean
          double x_mean = 0.0;
          for (int sample_index(0); sample_index < number_of_samples;sample_index++) {
            VImage image = (source_images[sample_index])[feature_index];
            x_mean += VGetPixel(image,band,row,column);
          }
          x_mean /= number_of_samples;

          // Calculate voxel standard deviation
          double x_sd = 0.0;
          for (int sample_index(0); sample_index < number_of_samples;sample_index++) {
            VImage image = (source_images[sample_index])[feature_index];
            double diff = VGetPixel(image,band,row,column)-x_mean;
            x_sd  += diff * diff;
          }
          x_sd = sqrt(x_sd/((double) (number_of_samples-1)));

          // Now we calculate the correlation
          double sum = 0.0;
          for (int sample_index(0); sample_index < number_of_samples;sample_index++) {
            VImage image = (source_images[sample_index])[feature_index];
            sum += (VGetPixel(image,band,row,column) - x_mean) * (y[sample_index] - y_mean);
          }

          double correlation = sum / ((number_of_samples - 1) * x_sd * y_sd);
          if (band == 10 && row == 40 && column == 26)
            cerr << "x_mean=" << x_mean << " x_sd=" << x_sd << " r=" << correlation << " cov=" << sum << endl;
          VPixel(dest,band,row,column,VFloat) = correlation;
          VPixel(z_dest,band,row,column,VFloat) = (artanh(correlation) - artanh(0)) * sqrt( number_of_samples - 3 );
        }
      }
    }

    VSetAttr(VImageAttrList(dest),"name",NULL,VStringRepn,"Correlation");
    VSetAttr(VImageAttrList(z_dest),"name",NULL,VStringRepn,"Correlation z");
    VAppendAttr(out_list,"image",NULL,VImageRepn,dest);
    VAppendAttr(out_list,"image",NULL,VImageRepn,z_dest);
  }
  VHistory(VNumber(program_options),program_options,argv[0],&attribute_list,&out_list);
  VWriteFile(out_file, out_list);

  lipsia_gettime(&end);

  long long int execution_time = (end.tv_sec * 1e9 + end.tv_nsec) - (start.tv_sec * 1e9 + start.tv_nsec);
  cout << "Execution time: " << execution_time / 1e9 << "s" << endl;

  
  delete[] source_images;
}

