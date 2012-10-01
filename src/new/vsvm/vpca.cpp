/**
 * @file vpca.cpp
 * 
 * PCA - Principal Component Analysis
 *
 * @author Tilo Buschmann
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
#define BOOST_DISABLE_ASSERTS
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
using std::cout;
using std::endl;
using std::setw;
using std::vector;
using std::map;
using std::string;
using std::ofstream;
using boost::assign::map_list_of;

extern "C" void getLipsiaVersion(char*,size_t);

#ifdef _OPENMP
#include <omp.h>
#endif /*_OPENMP*/

int main (int argc,char *argv[]) {
  /**************************
   * Initialise Vista Stuff *
   **************************/

  // Output program name and version
  char version[100];
  getLipsiaVersion(version, sizeof(version));
  cerr << argv[0] << " V" << version << endl;

  // Parse command line parameters
  static VArgVector input_filenames1,input_filenames2;
  VBoolean  do_permutations = false;
  VBoolean  do_pca          = false;

  FILE *out_file;

  static VOptionDescRec program_options[] = {
    {"in1",           VStringRepn,  0, &input_filenames1,VRequiredOpt, NULL, "Input files (class 1)" },
    {"in2",           VStringRepn,  0, &input_filenames2,VRequiredOpt, NULL, "Input files (class 2)" },
    {"pca",           VBooleanRepn, 1, &do_pca,          VOptionalOpt, NULL, "Whether to do a pca before the svm"},
    {"permutate",     VBooleanRepn, 1, &do_permutations, VOptionalOpt, NULL, "Calculate permutation based z scores"}
  };
  VParseFilterCmd(VNumber (program_options),program_options,argc,argv,NULL,&out_file);
  
  VArgVector  input_filenames[2] = {input_filenames1,input_filenames2};

  /*******************************************
   * Read image files and extract image data *
   *******************************************/

  int number_of_class_samples[2]    = {input_filenames1.number, input_filenames1.number};
  int number_of_samples             = input_filenames1.number+input_filenames1.number;

  VImage source_images[number_of_samples];

  long int number_of_features = 0;
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
      int this_number_of_features = VImageNPixels(source_images[sample_position]);
      if (0 == number_of_features) {
        number_of_features = this_number_of_features;
      } else if (number_of_features != this_number_of_features) {
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
          if (VGetPixel(source_images[sample_index],band,row,column) != 0.0) {
            voxel_is_empty[band][row][column] = false;
          }
        }
        if (!voxel_is_empty[band][row][column])
          used_voxels++;
      }
    }
  }
    
  cerr << "  Using " << used_voxels << " of " << number_of_features << " features in the images." << endl;
  matrix_2d sample_features(boost::extents[number_of_samples][used_voxels]);

  for(int sample_index(0); sample_index < number_of_samples; sample_index++) {
    int feature_index(0);
    for(int band(0); band < number_of_bands; band++) {
      for(int row(0); row < number_of_rows; row++) {
        for(int column(0); column < number_of_columns; column++) {
          if (!voxel_is_empty[band][row][column]) {
            sample_features[sample_index][feature_index] = VGetPixel(source_images[sample_index],band,row,column);
            feature_index++;
          }
        }
      }
    }
  }
  cerr << "done." << endl;

  /***************
   * Conduct PCA *
   ***************/

  matrix_2d X;

  if (do_pca) {
    cerr << "Conducting PCA ... ";
    PrComp result = PCA::prcomp(sample_features);

    X = result.getX();
    cerr << "done." << endl;
  } else {
    X = sample_features;
  }


  cerr << "Conducting SVM ... ";
  MriSvm mrisvm(X,classes,number_of_samples,result.getP());
  // Scale features
  mrisvm.scale();
  // Get weights
  boost::multi_array<double,1> weights(boost::extents[result.getP()]);
  mrisvm.train_weights(weights);
  cerr << "done." << endl;

  boost::multi_array<double,1> voxel_weights(boost::extents[used_voxels]);
  if (do_pca) {
    // Convert PC-Weights to Voxel-Weights
    result.invert(weights,voxel_weights);
  } else {
    voxel_weights = 
  }


  /****************
   * Permutations *
   ****************/
  boost::multi_array<double, 2> permutated_weights;
  boost::multi_array<double,1> permutated_voxel_weights;
  int actual_permutations = 0;
  if (do_permutations) {
    cerr << "Calculating SVM of permutations ... ";
    permutations_array_type permutations;
    actual_permutations = SearchLight::generate_permutations(number_of_samples,2,10000,permutations);

    permutated_weights.resize(boost::extents[actual_permutations][result.getP()]);

    mrisvm.permutated_weights(permutated_weights,actual_permutations,permutations);
  
    permutated_voxel_weights.resize(boost::extents[actual_permutations]);

    cerr << "done." << endl;
  }

  /****************
   * File writing *
   ****************/
  int feature_index = 0;

  VImage dest             = VCreateImage(number_of_bands,number_of_rows,number_of_columns,VFloatRepn);
  VFillImage(dest,VAllBands,0);
  VCopyImageAttrs (source_images[0], dest);

  VImage permutation_dest = VCreateImage(number_of_bands,number_of_rows,number_of_columns,VFloatRepn);
  VFillImage(permutation_dest,VAllBands,0);
  VCopyImageAttrs(source_images[0], permutation_dest);

  cerr << "Writing output image ... ";
  boost::progress_display writing_progress(used_voxels);

//#pragma omp parallel for default(none) shared(writing_progress,number_of_bands,number_of_rows,number_of_columns,dest,permutation_dest,voxel_is_empty,feature_index) schedule(dynamic)
  for(int band = 0; band < number_of_bands; band++) {
    for(int row(0); row < number_of_rows; row++) {
      for(int column(0); column < number_of_columns; column++) {
        if (!voxel_is_empty[band][row][column]) {
          /****************
           * Write weight *
           ****************/
          VPixel(dest,band,row,column,VFloat) = voxel_weights[feature_index];
         
          if (do_permutations) {
            /*****************
             * Write z-score *
             *****************/
            //struct timespec start,inversion,end;
            //clock_gettime(CLOCK_MONOTONIC,&start);
            result.invert_permutation(permutated_voxel_weights, permutated_weights, feature_index, actual_permutations);
            //clock_gettime(CLOCK_MONOTONIC,&inversion);

            double sum = 0.0;
            for (int i(0); i < actual_permutations; i++)
              sum += permutated_voxel_weights[i];

            double mean = sum / actual_permutations;
            double sd = 0.0;
            for (int i(0); i < actual_permutations; i++) {
              double diff = mean - permutated_voxel_weights[i];
              sd += diff * diff;
            }
            sd = sqrt(sd/(actual_permutations - 1));
            double z = (voxel_weights[feature_index] - mean) / sd;
            //clock_gettime(CLOCK_MONOTONIC,&end);
            //long long int execution_time_all = (end.tv_sec * 1e9 + end.tv_nsec) - (start.tv_sec * 1e9 + start.tv_nsec);
            //long long int execution_time_inv = (inversion.tv_sec * 1e9 + inversion.tv_nsec) - (start.tv_sec * 1e9 + start.tv_nsec);
            //long long int execution_time_rest = (end.tv_sec * 1e9 + end.tv_nsec) - (inversion.tv_sec * 1e9 + inversion.tv_nsec);
            //cout << "Execution time: " << execution_time_all / 1e6 << "ms" << " Inversion: " << execution_time_inv / 1e6 << "ms" << " Rest:" << execution_time_rest / 1e6 << "ms" << endl;

            //double z = 0.0;
            VPixel(permutation_dest,band,row,column,VFloat) = z;
          }

          feature_index++;
//#pragma omp critical
          ++writing_progress;
        }
      }
    }
  }
  cerr << "done." << endl;
  VSetAttr(VImageAttrList(dest),"name",NULL,VStringRepn,"PCA SVM Weights");
  VAttrList out_list = VCreateAttrList();
  VAppendAttr(out_list,"image",NULL,VImageRepn,dest);
   
  if (do_permutations) { 
    VSetAttr(VImageAttrList(permutation_dest),"name",NULL,VStringRepn,"PCA SVM Z");
    VAppendAttr(out_list,"image",NULL,VImageRepn,permutation_dest);
  }

  //VHistory(VNumber(program_options),program_options,argv[0],&attribute_list,&out_list);
  VWriteFile(out_file, out_list);

}

