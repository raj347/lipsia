/**
 * @file vsvm.cpp
 * 
 * SVM - Support Vector Machine
 * 
 * Usage:
 *  vsvm -in1 class1samples.v -in2 class2samples.v -out svm.v [-scale] [-pca] [-saveperm] [-nperm number of permutations] [-j nprocs] [svm options] [-permfile permutations.v]
 *
 *  options:
 *    -in1 class1samples.v
 *      Input files (class 1)
 *    -in2 class2samples.v
 *      Input files (class 2)
 *    -scale
 *      Whether to scale data
 *    -pca
 *      Wheter to process data with a principal component analysis
 *    -saveperm
 *      Whether to save permutations to output file
 *    -nperm
 *      Number of permutations (default: 0)
 *    -j nprocs
 *      number of processors to use, '0' to use all
 *    -permfile permutations.v
 *      store permutations in permutations.v instead of svm.v (requires -saveperm to have an effect)
 *
 *  svm options:
 *    -svm_cache_size cache_size
 *      cache size parameter (in MByte)
 *    -svm_eps epsilon
 *      epsilon parameter (stopping criteria)
 *    -svm_C C
 *      C parameter (for C_SVC svm type)
 *
 * @author Tilo Buschmann
 */

// C++ header
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>

// C header
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// GSL
#include <gsl/gsl_cdf.h>

// Boost header
//#define BOOST_DISABLE_ASSERTS
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
#include "PCA.h"
#include "MriSvm.h"

#define DEFAULT_VSVM_IMAGE_CLASS 0

using std::cerr;
using std::endl;
using std::setw;
using std::vector;
using std::string;
using std::stringstream;

extern "C" void getLipsiaVersion(char*,size_t);

struct xyz_coords {
  int band;
  int row;
  int column;
};

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
  if (number_of_cores == 1)
    cerr << "Using 1 core" << endl;
  else
    cerr << "Using " << number_of_cores << " cores" << endl;
  omp_set_num_threads(number_of_cores);
}
#endif /*OPENMP */

int main (int argc,char *argv[]) {
  /**************************
   * Initialise Vista Stuff *
   **************************/

  // Output program name and version
  /*
  char version[100];
  getLipsiaVersion(version, sizeof(version));
  cerr << argv[0] << " V" << version << endl;
  */
  cerr << argv[0] << " V2.2.1 [62-g0ab8cd2]" << endl;

  // Parse command line parameters
  VArgVector  input_filenames1,input_filenames2;
  VString     perm_filename = NULL;
  VBoolean    do_scale      = false;
  VBoolean    do_pca        = false;
  VBoolean    save_perms    = false;
  VBoolean    paired        = false;
  VShort      nproc         = 4;
  VLong       nperm         = 0;

  FILE *out_file;

  VDouble svm_cache_size  = DEFAULT_MRISVM_CACHE_SIZE;
  VDouble svm_eps         = DEFAULT_MRISVM_EPS;
  VDouble svm_C           = DEFAULT_MRISVM_C;

  static VOptionDescRec program_options[] = {
    {"in1",           VStringRepn,  0, &input_filenames1, VRequiredOpt, NULL, "Input files (class 1)" },
    {"in2",           VStringRepn,  0, &input_filenames2, VRequiredOpt, NULL, "Input files (class 2)" },
    {"scale",         VBooleanRepn, 1, &do_scale,         VOptionalOpt, NULL, "Whether to scale data"},
    {"pca",           VBooleanRepn, 1, &do_pca,           VOptionalOpt, NULL, "Whether to do a pca before the svm"},
    {"saveperm",      VBooleanRepn, 1, &save_perms,       VOptionalOpt, NULL, "Whether to store permutations" },
    {"permfile",      VStringRepn,  1, &perm_filename,    VOptionalOpt, NULL, "File to store permutations (if extra file)" },
    {"j",             VShortRepn,   1, &nproc,            VOptionalOpt, NULL, "number of processors to use, '0' to use all" },
    {"nperm",         VLongRepn,    1, &nperm,            VOptionalOpt, NULL, "number of permutations to generate (default: 0)" },
    {"paired",        VBooleanRepn, 1, &paired,           VOptionalOpt, NULL, "Conduct a paired test" },
    {"svm_cache_size",VDoubleRepn,  1, &svm_cache_size,   VOptionalOpt, NULL, "SVM cache size parameter (in MByte)" },
    {"svm_eps",       VDoubleRepn,  1, &svm_eps,          VOptionalOpt, NULL, "SVM eps parameter (stopping criteria)" },
    {"svm_C",         VDoubleRepn,  1, &svm_C,            VOptionalOpt, NULL, "SVM C parameter (for C_SVC svm type)" }
  };
  VParseFilterCmd(VNumber (program_options),program_options,argc,argv,NULL,&out_file);

  if (perm_filename != NULL) {
    cerr << "Filename: " << perm_filename << endl;
  }
  
  bool  do_permutations = (nperm > 0);
  
  VArgVector  input_filenames[2] = {input_filenames1,input_filenames2};

  /* Configure SVM parameters */

  struct svm_parameter parameter = MriSvm::get_default_parameters();
  parameter.svm_type    = MriSvm::C_SVC;
  parameter.kernel_type = MriSvm::LINEAR;

  parameter.cache_size  = svm_cache_size;
  parameter.eps         = svm_eps;
  parameter.C           = svm_C;

#ifdef _OPENMP
  /* Configure multi processing */
  configure_omp(nproc); 
#endif /*OPENMP */

  /*******************************************
   * Read image files and extract image data *
   *******************************************/

  int number_of_class_samples[2]    = { input_filenames1.number, input_filenames2.number };
  int number_of_samples             = input_filenames1.number+input_filenames2.number;

  VImage source_images[number_of_samples];

  long int number_of_voxels = 0;
  VAttrList attribute_list;

  cerr << "Reading Image Files ... " << endl;

  vector <double> classes(number_of_samples);
  int sample_position = 0;

  for (int current_class(0); current_class <= 1;current_class++) {
    cerr << "  Reading class " << current_class + 1 << " image files" << endl << "  ";
    for (int file_no(0); file_no < number_of_class_samples[current_class] ; file_no++) {
      source_images[sample_position]  = NULL;
      classes[sample_position]        = current_class;

      /*******************
       * Read image file *
       *******************/
      VStringConst input_filename = ((VStringConst *) input_filenames[current_class].vector)[file_no];
      cerr << input_filename << "(" << classes[sample_position] + 1 << ") ";
      FILE *input_file          = VOpenInputFile(input_filename, TRUE);
      attribute_list  = VReadFile(input_file, NULL);
      fclose(input_file);

      /**********************
       * Analyse attributes *
       **********************/

      if(!attribute_list)
        VError("Error reading image");

      VAttrListPosn position;
      for (VFirstAttr(attribute_list, &position); VAttrExists(&position); VNextAttr(&position)) {
        if (VGetAttrRepn(&position) != VImageRepn)
          continue;
        VImage image;
        VGetAttrValue(&position,NULL,VImageRepn,&image);

        source_images[sample_position] = image;
        break;
      }

      if (source_images[sample_position] == NULL) 
        VError("No input image found");

      // Get number of features (i.e. bands * rows * columns)
      long int this_number_of_voxels = VImageNPixels(source_images[sample_position]);
      if (0 == number_of_voxels) {
        number_of_voxels = this_number_of_voxels;
      } else if (number_of_voxels != this_number_of_voxels) {
        VError("Error: Number of features differs from number of features in previous pictures.");
      }
      sample_position++;
    }
    cerr << endl;
  }
  cerr << "done." << endl << endl;

  /*****************************
   * Convert to usable format  *
   *****************************/

  cerr << "Converting and filtering data ... " << endl;
  int used_voxels = 0;

  int number_of_bands   = VImageNBands(source_images[0]);
  int number_of_rows    = VImageNRows(source_images[0]);
  int number_of_columns = VImageNColumns(source_images[0]);

  bool voxel_is_empty[number_of_bands][number_of_rows][number_of_columns];
  for(int band(0); band < number_of_bands; band++) {
    for(int row(0); row < number_of_rows; row++) {
      for(int column(0); column < number_of_columns; column++) {
        voxel_is_empty[band][row][column] = true;
        for(int sample_index(0); sample_index < number_of_samples; sample_index++) {
          float voxel = VGetPixel(source_images[sample_index],band,row,column);
          if ((voxel != 0.0) && !(isnan(voxel))) {
            voxel_is_empty[band][row][column] = false;
          }
        }
        if (!voxel_is_empty[band][row][column])
          used_voxels++;
      }
    }
  }
    
  cerr << "  Using " << used_voxels << " of " << number_of_voxels << " voxels in the images." << endl;
  matrix_2d sample_features(boost::extents[number_of_samples][used_voxels]);

  boost::multi_array<struct xyz_coords, 1> feature_coords(boost::extents[used_voxels]);

  for(int sample_index(0); sample_index < number_of_samples; sample_index++) {
    int feature_index(0);
    for(int band(0); band < number_of_bands; band++) {
      for(int row(0); row < number_of_rows; row++) {
        for(int column(0); column < number_of_columns; column++) {
          if (!voxel_is_empty[band][row][column]) {
            //cerr << "f=" << feature_index << " (" << band << "/" << row << "/" << column << ")" << endl;
            sample_features[sample_index][feature_index] = VGetPixel(source_images[sample_index],band,row,column);
            feature_coords[feature_index].band    = band;
            feature_coords[feature_index].row     = row;
            feature_coords[feature_index].column  = column;
            feature_index++;
          }
        }
      }
    }
  }
  cerr << "done." << endl;

  if (paired) {
    int pairs = number_of_samples / 2;
    // Generate distances and stuff
    for(int pair_loop(0); pair_loop < pairs; pair_loop++) {
      cerr << "Difference between " << pair_loop << "(" << ((VStringConst *) input_filenames[0].vector)[pair_loop] << ") and " << (pair_loop + pairs) << "(" << ((VStringConst *) input_filenames[1].vector)[pair_loop] << ")" << endl;
      for (int feature_index = 0; feature_index < used_voxels; feature_index++) {
        double diff = sample_features[pair_loop + pairs][feature_index] - sample_features[pair_loop][feature_index];
        sample_features[pair_loop + pairs][feature_index] = diff;
        sample_features[pair_loop][feature_index] = -1.0 * diff;
      }
    }
  }

  
  /* Prepare output images (as long as we have the source images) */
  VImage dest = VCreateImage(number_of_bands,number_of_rows,number_of_columns,VFloatRepn);
  VFillImage(dest,VAllBands,0);
  VCopyImageAttrs (source_images[0], dest);

  VImage p_dest = VCreateImage(number_of_bands,number_of_rows,number_of_columns,VFloatRepn);
  VFillImage(p_dest,VAllBands,0);
  VCopyImageAttrs(source_images[0], p_dest);

  /* Now I can free the VImage data */
  for(int sample_index(0); sample_index < number_of_samples; sample_index++) {
    VDestroyImage(source_images[sample_index]);
    source_images[sample_index] = NULL;
  }

  matrix_2d svm_input;                      // to be used as SVM input
  PrComp    *pca_result             = NULL; // PCA results, might be needed later
  int       number_of_svm_features  = 0;    // actual number of features used as SVM input
  if (do_pca) {
    /***************
     * Conduct PCA *
     ***************/
    cerr << "Conducting PCA ... ";
    struct timespec start,end;
    clock_gettime(CLOCK_MONOTONIC,&start);
    pca_result = PCA::prcomp(sample_features);
    clock_gettime(CLOCK_MONOTONIC,&end);

    long long int execution_time = (end.tv_sec * 1e9 + end.tv_nsec) - (start.tv_sec * 1e9 + start.tv_nsec);
    cerr << "Execution time: " << execution_time / 1e9 << "s" << endl;
    cerr << "done." << endl;
    number_of_svm_features = pca_result->getP();

    matrix_2d X = pca_result->getX();

    const size_t* shape = X.shape();
    std::vector<size_t> ex;
    ex.assign(shape,shape+X.num_dimensions());
    svm_input.resize(ex);
    svm_input = X;

    cerr << "done." << endl;
  } else {
    const size_t* shape = sample_features.shape();
    std::vector<size_t> ex;
    ex.assign(shape,shape+sample_features.num_dimensions());
    svm_input.resize(ex);
    svm_input = sample_features;

    number_of_svm_features = used_voxels;
  }

  /***************
   * Conduct SVM *
   ***************/

  cerr << "Conducting SVM ... ";
  struct timespec start,end;
  clock_gettime(CLOCK_MONOTONIC,&start);

  MriSvm mrisvm(svm_input, classes, number_of_samples, number_of_svm_features);
  mrisvm.set_parameters(parameter);

  // Scale features
  if (do_scale) {
    cerr << "SKALIERE!" << endl;
    mrisvm.scale();
  } else {
    cerr << "Skaliere nicht." << endl;
  }

  // Get weights
  boost::multi_array<float, 1> weights;
  mrisvm.train_weights(weights);
  //cerr << "CV" << mrisvm.cross_validate(2) << endl;
  clock_gettime(CLOCK_MONOTONIC,&end);

  long long int execution_time = (end.tv_sec * 1e9 + end.tv_nsec) - (start.tv_sec * 1e9 + start.tv_nsec);
  cerr << "Execution time: " << execution_time / 1e9 << "s" << endl;
  cerr << "done." << endl;

  // FIXME invert inplace
  boost::multi_array<float, 1> voxel_weights(boost::extents[used_voxels]);
  if (do_pca) {
    // Convert PC-Weights to Voxel-Weights
    pca_result->invert(weights,voxel_weights);
  } else {
    voxel_weights = weights;
  }

  /****************
   * Permutations *
   ****************/
  gsl_matrix_float* voxel_permutation_weights = NULL;
  gsl_matrix_float* permutation_weights       = NULL;
  
  int actual_permutations = 0;
  permutations_array_type permutations;

  if (do_permutations) {
    cerr << "Calculating SVM of permutations ... ";
    if (paired)
      actual_permutations = MriSvm::generate_paired_permutations(number_of_samples, nperm, permutations);
    else
      actual_permutations = MriSvm::generate_permutations(number_of_samples, nperm, permutations);

    cerr << "Actual number of permutations: " << actual_permutations << endl;
    permutation_weights = mrisvm.permutated_weights( actual_permutations, permutations );
    
    cerr << "done." << endl;

    if (do_pca) {
      // invert matrix
      cerr << "Inverting Matrix... ";
      voxel_permutation_weights = pca_result->invert_matrix( permutation_weights, used_voxels, actual_permutations);
      cerr << "done." << endl;
      gsl_matrix_float_free(permutation_weights);
    } else {
      voxel_permutation_weights = permutation_weights;
    }
  }

  /****************
   * File writing *
   ****************/
  cerr << "Generating weights image ... ";
  boost::progress_display writing_progress(used_voxels);
  
  VAttrList out_list = VCreateAttrList();

  vector<float> voxel_specific_permutations(actual_permutations);

#pragma omp parallel for default(none) shared(feature_coords, dest, p_dest, writing_progress,  do_permutations, voxel_permutation_weights, actual_permutations, voxel_weights, used_voxels, cerr) firstprivate(voxel_specific_permutations) schedule(dynamic)
  for (int feature_index = 0; feature_index < used_voxels; feature_index++) {
    int band    = feature_coords[feature_index].band;
    int row     = feature_coords[feature_index].row;
    int column  = feature_coords[feature_index].column;

    //cerr << "f=" << feature_index << " (" << band << "/" << row << "/" << column << ")" << endl;

    /****************
     * Write weight *
     ****************/
    VPixel(dest,band,row,column,VFloat) = voxel_weights[feature_index];

    if (do_permutations) {
      /*****************
       * Write p value *
       *****************/
      for (int i = 0; i < actual_permutations; i++) {
        voxel_specific_permutations[i] = gsl_matrix_float_get(voxel_permutation_weights, i, feature_index);
      }

      std::sort( voxel_specific_permutations.begin(), voxel_specific_permutations.end());

      int weight_index(0);
      for (; (weight_index < actual_permutations) && (voxel_specific_permutations[weight_index] < voxel_weights[feature_index]); weight_index++) {
      }

      float p = (float) weight_index / (float) actual_permutations;

      // Correct for two sided test
      if (p > 0.5)
        p = 1 - p;

      p *= 2;

      // Correction for 0, do not double!
      if ((weight_index == 0) || (weight_index == actual_permutations))
        p = 1.0 / (float) actual_permutations;

      VPixel(p_dest,band,row,column,VFloat) = -log10(p);
    }
    ++writing_progress;
  }

  VSetAttr(VImageAttrList(dest),"name",NULL,VStringRepn,"PCA SVM Weights");
  VAppendAttr(out_list,"image",NULL,VImageRepn,dest);
   
  if (do_permutations) { 
    VSetAttr(VImageAttrList(p_dest),"name",NULL,VStringRepn,"PCA SVM non-parametric p");
    VAppendAttr(out_list,"image",NULL,VImageRepn,p_dest);
  }
  
  cerr << "done." << endl;

  if (do_permutations && save_perms) {
    cerr << "Generating permutation images." << endl;

    VAttrList perm_out_list = VCreateAttrList();

    boost::progress_display permutation_progress(actual_permutations);
    for(int permutation_loop(0); permutation_loop < actual_permutations; permutation_loop++) {
      int feature_index = 0;

      VImage permutation_dest = VCreateImage(number_of_bands,number_of_rows,number_of_columns,VFloatRepn);
      VFillImage(permutation_dest,VAllBands,0);
      VCopyImageAttrs (dest, permutation_dest);

      for(int band = 0; band < number_of_bands; band++) {
        for(int row(0); row < number_of_rows; row++) {
          for(int column(0); column < number_of_columns; column++) {
            if (!voxel_is_empty[band][row][column]) {
              VPixel(permutation_dest,band,row,column,VFloat) = gsl_matrix_float_get( voxel_permutation_weights, permutation_loop ,feature_index);
              feature_index++;
            }
          }
        }
      }


      stringstream permutation_text;
      permutation_text << permutations[permutation_loop][0];
      for (int sample_loop(1); sample_loop < number_of_samples; sample_loop++) {
        permutation_text << "," << permutations[permutation_loop][sample_loop];
      }

      VSetAttr(VImageAttrList(permutation_dest),"permutation",NULL,VStringRepn,permutation_text.str().c_str());
      VSetAttr(VImageAttrList(permutation_dest),"name",NULL,VStringRepn,"SVM Permutation");

      if (perm_filename != NULL) {
        VAppendAttr(perm_out_list,"image",NULL,VImageRepn,permutation_dest);
      } else {
        VAppendAttr(out_list,"image",NULL,VImageRepn,permutation_dest);
      }
      ++permutation_progress;
    }

    if (perm_filename != NULL) {
      cerr << "Writing extra file" << endl;
      FILE  *perm_file  = fopen(perm_filename,"w");
      cerr << "File: " << perm_file << endl;
      VWriteFile(perm_file, perm_out_list);
      fclose(perm_file);
    }

    cerr << "done." << endl;
  }

  cerr << "Saving to disk ...";
  VWriteFile(out_file, out_list);
  cerr << "done." << endl;
}

