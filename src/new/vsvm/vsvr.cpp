/**
 * @file vsvr.cpp
 * 
 * PCA + SVR
 *
 * @author Tilo Buschmann
 * @date 27.8.2012
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

int main (int argc,char *argv[]) {
  /**************************
   * Initialise Vista Stuff *
   **************************/

  // Output program name and version
  char version[100];
  getLipsiaVersion(version, sizeof(version));
  cerr << argv[0] << " V" << version << endl;

  // Parse command line parameters
  static VArgVector input_filenames;
  VBoolean  do_permutations = false;
  VString y_file            = NULL;

  FILE *out_file;

  static VOptionDescRec program_options[] = {
    {"in",        VStringRepn,  0, &input_filenames,  VRequiredOpt, NULL, "Input files" },
    {"y",         VStringRepn,  1, &y_file,           VRequiredOpt, NULL, "File containing the dependent values" },
    {"permutate", VBooleanRepn, 1, &do_permutations,  VOptionalOpt, NULL, "Calculate permutation based z scores"}
  };
  VParseFilterCmd(VNumber (program_options),program_options,argc,argv,NULL,&out_file);
  
  int number_of_samples             = input_filenames.number;
  
  /*************************
   * Read Correlation File *
   *************************/
  vector <double> y(number_of_samples);

  cerr << "Reading predictee variable file \"" << y_file << "\" ... ";
  ifstream input(y_file);

  double f;
  int sample_index;
  for (sample_index = 0, f = 0.0; (input >> f) && (sample_index < number_of_samples); sample_index++) {
    //cerr << "Read value " << f << " at sample_index " << sample_index << endl;
    y[sample_index] = f;
  }
  cerr << "done." << endl;

  /*******************************************
   * Read image files and extract image data *
   *******************************************/


  VImage source_images[number_of_samples];

  long int number_of_features = 0;
  VAttrList attribute_list;

  cerr << "Reading Image Files ... " << endl;

  for (int sample_index(0); sample_index < number_of_samples;sample_index++) {
    source_images[sample_index]  = NULL;

    /*******************
     * Read image file *
     *******************/
    VStringConst input_filename = ((VStringConst *) input_filenames.vector)[sample_index];
    cerr << input_filename << "(" << y[sample_index] << ")  ";
    FILE *input_file = VOpenInputFile(input_filename, TRUE);
    attribute_list   = VReadFile(input_file, NULL);
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

      source_images[sample_index] = image;
      break;
    }

    if (source_images[sample_index] == NULL) 
      VError("No input image found");

    // Get number of features (i.e. bands * rows * columns)
    int this_number_of_features = VImageNPixels(source_images[sample_index]);
    if (0 == number_of_features) {
      number_of_features = this_number_of_features;
    } else if (number_of_features != this_number_of_features) {
      VError("Error: Number of features differs from number of features in previous pictures.");
    }
  }
  cerr << endl << "done." << endl << endl;

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

  cerr << "Converting Data into PCA Format ... ";
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
  cerr << "Conducting PCA ... ";
  PrComp result = PCA::prcomp(sample_features);
  matrix_2d X = result.getX();
  cerr << "done." << endl;

  cerr << "Conducting SVR ... ";
  MriSvm mrisvm(result.getX(),y,number_of_samples,result.getP());
  mrisvm.set_svm_type(MriSvm::EPSILON_SVR);
  // Scale features
  mrisvm.scale();
  // Get weights
  boost::multi_array<double,1> weights(boost::extents[result.getP()]);
  mrisvm.train_weights(weights);
  cerr << "done." << endl;
  
  // Convert PC-Weights to Voxel-Weights
  boost::multi_array<double,1> voxel_weights(boost::extents[used_voxels]);
  result.invert(weights,voxel_weights);

  /****************
   * Permutations *
   ****************/
  boost::multi_array<double, 2> permutated_weights;
  boost::multi_array<double,1>  permutated_voxel_weights;
  int actual_permutations = 0;

  if (do_permutations) {
    cerr << "Calculating SVR of permutations ... ";
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
  for(int band(0); band < number_of_bands; band++) {
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
            result.invert_permutation(permutated_voxel_weights, permutated_weights, feature_index, actual_permutations);

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
            VPixel(permutation_dest,band,row,column,VFloat) = z;
          }

          feature_index++;
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

