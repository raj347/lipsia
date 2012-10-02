/**
 * @file vsl.cpp
 * 
 * Wrapper around SearchLight class to load samples from Vista images, process
 * SearchlightSVM and then write result into another Vista file
 *
 * Usage:
 *  vsl -in1 class1samples.v -in2 class2samples.v [-radius radius] [-scale] [-permutate] [-saveperm] [-nperm number of permutations] [-j nprocs] [svm options]
 *
 *  options:
 *    -in1 class1samples.v
 *      Input files (class 1)
 *    -in2 class2samples.v
 *      Input files (class 2)
 *    -radius radius
 *      Searchlight Radius (in mm)
 *    -scale
 *      Whether to scale data
 *    -permutate
 *      Whether to permutate data and genrate a z map (takes some time)
 *    -saveperm 
 *      Whether to save permutations to output file
 *    -nperm
 *      Number of permutations (default: 100)
 *    -j nprocs
 *      number of processors to use, '0' to use all
 *
 *  svm options:
 *    -svm_type C_SVC | NU_SVC
 *      Type parameter
 *    -svm_kernel LINEAR | POLY | RBF | SIGMOID | PRECOMPUTED
 *      Kernel parameter
 *    -svm_degree degree
 *      degree parameter (for POLY kernel)
 *    -svm_gamma gamma
 *      gamma parameter (for POLY,RBF, and SIGMOID kernels)
 *    -svm_coef0 coef0
 *      coef0 parameter (for POLY and SIGMOID)
 *    -svm_cache_size cache_size
 *      cache size parameter (in MByte)
 *    -svm_eps epsilon
 *      epsilon parameter (stopping criteria)
 *    -svm_C C
 *      C parameter (for C_SVC svm type)
 *    -svm_nu nu
 *      nu parameter (for NU_SVC svm type)
 *
 * @author Tilo Buschmann
 *
 */

// C++ header
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>
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

// Class header
#include "MriSvm.h"
#include "MriTypes.h"
#include "SearchLight.h"

// Some stuff I am using from boost and std lib, explicitly declared
using std::cerr;
using std::cout;
using std::endl;
using std::setw;
using std::vector;
using std::map;
using std::string;
using std::ofstream;
using std::stringstream;
using boost::assign::map_list_of;

extern "C" void getLipsiaVersion(char*,size_t);

#ifdef _OPENMP
#include <omp.h>
#endif /*_OPENMP*/

/**
 * Parses the command line parameter for the SVM Type
 * (C_SVC,NU_SVC,ONE_CLASS,EPSILON_SVR,NU_SVR) and converts it to the
 * respective enum value from MriSVM 
 *  
 *  @param[in]  svm_type C string with command line parameter for SVM type
 *
 *  @return enum-equivalent of svm type
 */
int parseSvmType(VString svm_type) {
  if (NULL == svm_type)
    return DEFAULT_MRISVM_SVM_TYPE;

  static map<string, MriSvm::SVM_TYPE> svm_type_map = map_list_of("C_SVC",      MriSvm::C_SVC)
                                                                 ("NU_SVC",     MriSvm::NU_SVC);

  map<string, MriSvm::SVM_TYPE>::iterator it;
  
  it = svm_type_map.find(svm_type);

  if (it == svm_type_map.end()) {
    return DEFAULT_MRISVM_SVM_TYPE;
  }

  return it->second;

}

/**
 *
 * Parses  the command line parameter for the SVM Kernel
 * (LINEAR,POLY,RBF,SIGMOID,PRECOMPUTED) and converts it to the respective enum
 * value from MriSVM 
 *   
 * @param[in]     svm_kernel_type C string with command line parameter for SVM kernel
 *
 * @return enum-equivalent of svm kernel
*/
int parseSvmKernelType(VString svm_kernel_type) {
  //cerr << "Kernel type to be parsed: " << svm_kernel_type << endl;
  if (NULL == svm_kernel_type)
    return DEFAULT_MRISVM_KERNEL_TYPE;

  static map<string, MriSvm::KERNEL_TYPE> svm_kernel_type_map = map_list_of("LINEAR",       MriSvm::LINEAR)
                                                                           ("POLY",         MriSvm::POLY)
                                                                           ("RBF",          MriSvm::RBF)
                                                                           ("SIGMOID",      MriSvm::SIGMOID)
                                                                           ("PRECOMPUTED",  MriSvm::PRECOMPUTED);

  map<string, MriSvm::KERNEL_TYPE>::iterator it;
  
  it = svm_kernel_type_map.find(svm_kernel_type);

  if (it == svm_kernel_type_map.end()) {
    cerr << "Did not find this kernel type" << endl;
    return DEFAULT_MRISVM_KERNEL_TYPE;
  }

  cerr << "Found kernel type: " << it->second << endl;
  return it->second;

}

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
  VDouble           radius          = DEFAULT_SEARCHLIGHT_RADIUS;

  VBoolean          do_scale        = false;
  VBoolean          do_permutations = false;
  VBoolean          save_perms      = false;

  VShort            nproc           = 4;
  VShort            nperm           = 100;
  static VArgVector input_filenames1,input_filenames2;

  /* 
   * SVM Parameters
   */
  VString           svm_type        = NULL;
  VString           svm_kernel_type = NULL;
  VLong             svm_degree      = DEFAULT_MRISVM_DEGREE;
  VDouble           svm_gamma       = DEFAULT_MRISVM_GAMMA;
  VDouble           svm_coef0       = DEFAULT_MRISVM_COEF0;
  VDouble           svm_cache_size  = DEFAULT_MRISVM_CACHE_SIZE;
  VDouble           svm_eps         = DEFAULT_MRISVM_EPS;
  VDouble           svm_C           = DEFAULT_MRISVM_C;
  VDouble           svm_nu          = DEFAULT_MRISVM_NU;

  FILE *out_file;

  static VOptionDescRec program_options[] = {
    {"in1",           VStringRepn, 0, &input_filenames1,VRequiredOpt, NULL, "Input files (class 1)" },
    {"in2",           VStringRepn, 0, &input_filenames2,VRequiredOpt, NULL, "Input files (class 2)" },
    {"radius",        VDoubleRepn, 1, &radius,          VOptionalOpt, NULL, "Searchlight Radius (in mm)" },
    {"scale",         VBooleanRepn,1, &do_scale,        VOptionalOpt, NULL, "Whether to scale data"},
    {"saveperms",     VBooleanRepn,1, &save_perms,      VOptionalOpt, NULL, "Whether to store permutations" },
    {"j",             VShortRepn,  1, &nproc,           VOptionalOpt, NULL, "number of processors to use, '0' to use all" },
    {"nperm",         VShortRepn,  1, &nperm,           VOptionalOpt, NULL, "number of permutations to generate" },
    {"permutate",     VBooleanRepn,1, &do_permutations, VOptionalOpt, NULL, "Calculate permutation based z scores"},
    {"svm_type",      VStringRepn, 1, &svm_type,        VOptionalOpt, NULL, "SVM Type parameter (C_SVC or NU_SVC)" },
    {"svm_kernel",    VStringRepn, 1, &svm_kernel_type, VOptionalOpt, NULL, "SVM Kernel parameter (LINEAR, POLY, RBF, SIGMOID, or PRECOMPUTED)" },
    {"svm_degree",    VLongRepn,   1, &svm_degree,      VOptionalOpt, NULL, "SVM degree parameter (for POLY kernel)" },
    {"svm_gamma",     VDoubleRepn, 1, &svm_gamma,       VOptionalOpt, NULL, "SVM gamma parameter (for POLY, RBF, and SIGMOID kernels)" },
    {"svm_coef0",     VDoubleRepn, 1, &svm_coef0,       VOptionalOpt, NULL, "SVM coef0 parameter (for POLY and SIGMOID kernels)" },
    {"svm_cache_size",VDoubleRepn, 1, &svm_cache_size,  VOptionalOpt, NULL, "SVM cache size parameter (in MByte)" },
    {"svm_eps",       VDoubleRepn, 1, &svm_eps,         VOptionalOpt, NULL, "SVM eps parameter (stopping criteria)" },
    {"svm_C",         VDoubleRepn, 1, &svm_C,           VOptionalOpt, NULL, "SVM C parameter (for C_SVC svm type)" },
    {"svm_nu",        VDoubleRepn, 1, &svm_nu,          VOptionalOpt, NULL, "SVM nu parameter (for NU_SVC svm type)" }
  };
  VParseFilterCmd(VNumber(program_options),program_options,argc,argv,NULL,&out_file);

  VArgVector  input_filenames[2] = {input_filenames1,input_filenames2};

  // Translate SVM parameter strings to enums
  int svm_type_parsed         = parseSvmType(svm_type);
  int svm_kernel_type_parsed  = parseSvmKernelType(svm_kernel_type);
  
  // I build my own SVM paramterset
  struct svm_parameter parameter = MriSvm::get_default_parameters();

  parameter.svm_type    = svm_type_parsed;
  parameter.kernel_type = svm_kernel_type_parsed;
  parameter.C           = svm_C;
  parameter.cache_size  = svm_cache_size;
  parameter.degree      = svm_degree;
  parameter.gamma       = svm_gamma;
  parameter.coef0       = svm_coef0;
  parameter.eps         = svm_eps;
  parameter.nu          = svm_nu;
  
  cerr << "Type: " << parameter.kernel_type << endl;

#ifdef _OPENMP
  // Take care of multiprocessing
  configure_omp(nproc); 
#endif /*OPENMP */

  /*************************************
   * Read image files and extract data *
   *************************************/

  int number_of_class_samples[2]    = {input_filenames1.number, input_filenames1.number};
  int number_of_samples             = input_filenames1.number+input_filenames1.number;

  int number_of_voxels              = 0;
  int number_of_features_per_voxel  = 0;

  vector<VImage> *source_images = new vector<VImage>[number_of_samples];

  VAttrList attribute_list;
  
  // Classes vector
  vector<double> classes(number_of_samples);

  int sample_position = 0;

  for (int current_class(0); current_class <= 1;current_class++) {
    cerr << "Reading class " << current_class + 1 << " image files" << endl;
    for (int file_no(0); file_no < number_of_class_samples[current_class] ; file_no++) {
      classes[sample_position] = current_class;
    
      /*******************
       * Read VIA file *
       *******************/
      VStringConst input_filename = ((VStringConst *) input_filenames[current_class].vector)[file_no];
      cerr << input_filename << "(" << classes[sample_position] << ")" << endl;
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
        source_images[sample_position].push_back(image);
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

      sample_position++;
    }
  }

  /*****************************
   * Convert to usable format  *
   *****************************/

  int number_of_bands   = VImageNBands(source_images[0].front());
  int number_of_rows    = VImageNRows(source_images[0].front());
  int number_of_columns = VImageNColumns(source_images[0].front());

  /*******************************************
   * Convert data into SearchlightSVM format *
   *******************************************/

  // Data array
  // (four dimensions: number_of_bands * number_of_rows * number_of_columns * number_of_features_per_voxel)
  sample_3d_array_type sample_features(boost::extents[number_of_samples][number_of_bands][number_of_rows][number_of_columns][number_of_features_per_voxel]);

  cerr << "Converting Data into SearchLightSvm Format" << endl;
  boost::progress_display convert_progress(number_of_samples);
  for(int sample_index(0); sample_index < number_of_samples; sample_index++) {
    ++convert_progress;

    BOOST_FOREACH(VImage image, source_images[sample_index]) {
      int feature_index = 0;
      for(int band(0); band < number_of_bands; band++) {
        for(int row(0); row < number_of_rows; row++) {
          for(int column(0); column < number_of_columns; column++) {
            sample_features[sample_index][band][row][column][feature_index] = VGetPixel(image,band,row,column);
          }
        }
      }
      feature_index++;
    }
  }

  /***********************************************
   * Find extension of voxels from first picture *
   ***********************************************/
  VString voxel_extension;
  if(VGetAttr(VImageAttrList(source_images[0].front()), "voxel", NULL, VStringRepn, &voxel_extension) != VAttrFound) {
    cerr << "Image does not have voxel attribute (giving the extension sizes)." << endl;
    exit(-1);
  }

  double extension_band,extension_row,extension_column;
  if (sscanf(voxel_extension,"%lf %lf %lf",&extension_band,&extension_row,&extension_column) != 3) {
    cerr << "Cannot parse voxel extension values. Need three double values." << endl;
    exit(-1);
  }

  /*****************************
   * Searchlight SVM - Engage! *
   *****************************/

  SearchLight sl(number_of_bands,
                 number_of_rows,
                 number_of_columns,
                 number_of_samples,
                 number_of_features_per_voxel,
                 sample_features,
                 classes,
                 extension_band,
                 extension_row,
                 extension_column);
  
  sl.set_parameters(parameter);
  
  /* Measure time */
  struct timespec start,end;
  clock_gettime(CLOCK_MONOTONIC,&start);

  if (do_scale) {
    sl.scale();
  }

  sample_validity_array_type validities = sl.calculate(radius);

  clock_gettime(CLOCK_MONOTONIC,&end);

  long long int execution_time = (end.tv_sec * 1e9 + end.tv_nsec) - (start.tv_sec * 1e9 + start.tv_nsec);
  cout << "Execution time: " << execution_time / 1e9 << "s" << endl;

  permutated_validities_type permutated_validities;
  permutations_array_type permutations;

  int actual_permutations  = 0;
  if (do_permutations) {
    clock_gettime(CLOCK_MONOTONIC,&start);
    permutated_validities.resize(boost::extents[number_of_bands][number_of_rows][number_of_columns][nperm]);

    actual_permutations = sl.calculate_permutations(permutated_validities,permutations,nperm,radius);
    cerr << "Actual permutations: " << actual_permutations << endl;

    clock_gettime(CLOCK_MONOTONIC,&end);
    long long int execution_time = (end.tv_sec * 1e9 + end.tv_nsec) - (start.tv_sec * 1e9 + start.tv_nsec);
    cout << "Execution time: " << execution_time / 1e9 << "s" << endl;
  }

  /*******************************
   * Save result into vista file *
   *******************************/

  cerr << "Preparing image data ...(" << number_of_bands << "/" << number_of_rows << "/" << number_of_columns << ")" << endl;

  VImage dest   = VCreateImage(number_of_bands,number_of_rows,number_of_columns,VFloatRepn);
  cerr << "Created images" << endl;
  VImage p_dest = VCreateImage(number_of_bands,number_of_rows,number_of_columns,VFloatRepn);
  cerr << "Created images" << endl;
  
  VFillImage(dest,  VAllBands,0);
  VFillImage(p_dest,VAllBands,0);
  
  cerr << "Filled with zeros" << endl;

  VCopyImageAttrs (source_images[0].front(), dest);
  VCopyImageAttrs (source_images[0].front(), p_dest);
  
  cerr << "Copied attributes" << endl;

  cerr << "starting" << endl;
  for(int band(0); band < number_of_bands; band++) {
    for(int row(0); row < number_of_rows; row++) {
      for(int column(0); column < number_of_columns; column++) {
        VPixel(dest,band,row,column,VFloat) = validities[band][row][column];

        if (do_permutations) {
          if (fabs(validities[band][row][column] - 0.0) < 1e-8) {
            VPixel(p_dest,band,row,column,VFloat) = 0.0;
          } else {

            double sum = 0.0;
            for (int i(0); i < actual_permutations; i++)
              sum += permutated_validities[band][row][column][i];


            double mean = sum / actual_permutations;
            double sd = 0.0;
            for (int i(0); i < actual_permutations; i++) {
              double diff = mean - permutated_validities[band][row][column][i];
              sd += diff * diff;
            }
            sd = sqrt(sd/(actual_permutations - 1));
            double z = (validities[band][row][column] - mean) / sd;

            VPixel(p_dest,band,row,column,VFloat) = z;
          }
        }
      }
    }
  }

  cerr << "done." << endl;
  cerr << "Writing to disk ...";
  VSetAttr(VImageAttrList(dest),"name",NULL,VStringRepn,"SearchlightSVM CV");
  VSetAttr(VImageAttrList(p_dest),"name",NULL,VStringRepn,"SearchlightSVM Z");
  VAttrList out_list = VCreateAttrList();
  VAppendAttr(out_list,"image",NULL,VImageRepn,dest);
  if (do_permutations) {
    VAppendAttr(out_list,"image",NULL,VImageRepn,p_dest);
  }

  if (do_permutations && save_perms) {
    for(int permutation_loop(0); permutation_loop < actual_permutations; permutation_loop++) {
      VImage dest = VCreateImage(number_of_bands,number_of_rows,number_of_columns,VFloatRepn);
      VFillImage(dest,VAllBands,0);
      VCopyImageAttrs (source_images[0].front(), dest);

      for(int band(0); band < number_of_bands; band++) {
        for(int row(0); row < number_of_rows; row++) {
          for(int column(0); column < number_of_columns; column++) {
            VPixel(dest,band,row,column,VFloat) = permutated_validities[band][row][column][permutation_loop];
          }
        }
      }
      // Add permutation to image
      stringstream permutation_text;

      permutation_text << permutations[permutation_loop][0];
      for (int sample_loop(1); sample_loop < number_of_samples; sample_loop++) {
        permutation_text << "," << permutations[permutation_loop][sample_loop];
      }

      VSetAttr(VImageAttrList(dest),"permutation",NULL,VStringRepn,permutation_text.str().c_str());
      VSetAttr(VImageAttrList(dest),"name",NULL,VStringRepn,"Searchlight Permutation");
      VAppendAttr(out_list,"image",NULL,VImageRepn,dest);
    }
  }

  VHistory(VNumber(program_options),program_options,argv[0],&attribute_list,&out_list);
  VWriteFile(out_file, out_list);
  cerr << "done." << endl;
  
  delete[] source_images;
  exit(0);
}

