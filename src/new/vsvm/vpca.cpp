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
  static VArgVector input_filenames;

  FILE *out_file;

  static VOptionDescRec program_options[] = {
    {"in",           VStringRepn, 0, &input_filenames, VRequiredOpt, NULL, "Input files" }
  };
  VParseFilterCmd(VNumber (program_options),program_options,argc,argv,NULL,&out_file);

  /*******************************************
   * Read image files and extract image data *
   *******************************************/

  int number_of_images = input_filenames.number;
  VImage source_images[number_of_images];

  long int number_of_features = 0;
  VAttrList attribute_list;

  cerr << "Reading Image Files" << endl;

  boost::progress_display file_progress(number_of_images);
  for(int i(0); i < number_of_images; i++) {
    ++file_progress;
    source_images[i] = NULL;

    /*******************
     * Read image file *
     *******************/

    VStringConst input_filename = ((VStringConst *) input_filenames.vector)[i];
    //cerr << setw(3) << i << ": Reading " << input_filename << endl;
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

      // Get name (i.e. what type of image this is)
      VString name;
      VGetAttr(VImageAttrList(image), "name", NULL, VStringRepn, &name);
      //cerr << "  Name: " << name << endl;

      source_images[i] = image;
      break;
    }

    if (source_images[i] == NULL) VError("No input image found");

    // Get number of features (i.e. bands * rows * columns)
    int this_number_of_features = VImageNPixels(source_images[i]);
    if (0 == number_of_features) {
      number_of_features = this_number_of_features;
    } else if (number_of_features != this_number_of_features) {
      VError("Error: Number of features differs from number of features in previous pictures.");
    }

  }

  /*****************************
   * Convert to usable format  *
   *****************************/

  vector <double> classes(number_of_images);

  int used_voxels = 0;

  int number_of_bands   = VImageNBands(source_images[0]);
  int number_of_rows    = VImageNRows(source_images[0]);
  int number_of_columns = VImageNColumns(source_images[0]);

  bool voxel_is_empty[number_of_bands][number_of_rows][number_of_columns];
  for(int band(0); band < number_of_bands; band++) {
    for(int row(0); row < number_of_rows; row++) {
      for(int column(0); column < number_of_columns; column++) {
        voxel_is_empty[band][row][column] = true;
        for(int sample_index(0); sample_index < number_of_images; sample_index++) {
          if (VGetPixel(source_images[sample_index],band,row,column) != 0.0) {
            voxel_is_empty[band][row][column] = false;
          }
        }
        if (!voxel_is_empty[band][row][column])
          used_voxels++;
      }
    }
  }

  cout << "Features=" << number_of_features << " Used voxels=" << used_voxels << endl;
  matrix_2d sample_features(boost::extents[number_of_images][used_voxels]);

  cerr << "Converting Data into PCA Format" << endl;
  boost::progress_display convert_progress(number_of_images);
  for(int sample_index(0); sample_index < number_of_images; sample_index++) {
    ++convert_progress;

    double image_class = DEFAULT_VSVM_IMAGE_CLASS;
    if(VGetAttr(VImageAttrList(source_images[sample_index]), "class", NULL, VDoubleRepn, &image_class) != VAttrFound) {
      cerr << "Image does not have class attribute. Using default value (" << DEFAULT_VSVM_IMAGE_CLASS << ")" << endl;
    }
    classes[sample_index] = image_class;
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

  /***************
   * Conduct PCA *
   ***************/

  PrComp result = PCA::prcomp(sample_features);

  matrix_2d X = result.getX();

  MriSvm mrisvm(result.getX(),classes,number_of_images,result.getP());

  // Scale features
  mrisvm.scale();

  // Get weights
  vector<double> weights = mrisvm.train_weights();

  // Convert PC-Weights to Voxel-Weights
  vector<double> voxel_weights = result.invert(weights);
  cout << voxel_weights.size() << endl;


  /****************
   * Permutations *
   ****************/
  permutations_array_type permutations;
  int actual_permutations = SearchLight::generate_permutations(number_of_images,2,10000,permutations);

  cerr << "Got all permutations" << endl;
  typedef vector<double> weight_type;
  vector<weight_type> permutated_weights = mrisvm.permutated_weights(actual_permutations,permutations);
  cerr << "Got all weights" << endl;
  cerr << permutated_weights.size() << endl;

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

  for(int band(0); band < number_of_bands; band++) {
    for(int row(0); row < number_of_rows; row++) {
      for(int column(0); column < number_of_columns; column++) {
        if (!voxel_is_empty[band][row][column]) {
          /****************
           * Write weight *
           ****************/
          VPixel(dest,band,row,column,VFloat) = voxel_weights[feature_index];
          
          /*****************
           * Write z-score *
           *****************/
          vector<double> permutated_voxel_weights = result.invert_permutation(permutated_weights, feature_index);

          double sum = 0.0;
          BOOST_FOREACH( double voxel_weight , permutated_voxel_weights) {
            sum += voxel_weight;
          }
          double mean = sum / permutated_voxel_weights.size();
          double sd = 0.0;
          BOOST_FOREACH( double voxel_weight , permutated_voxel_weights) {
            double diff = mean - voxel_weight;
            sd += diff * diff;
          }
          sd = sqrt(sd/(permutated_voxel_weights.size() - 1));
          double z = (voxel_weights[feature_index] - mean) / sd;

          //double z = 0.0;
          VPixel(permutation_dest,band,row,column,VFloat) = z;

          feature_index++;
        }
      }
    }
  }
  VSetAttr(VImageAttrList(dest),"name",NULL,VStringRepn,"PCA SVM Weights");
  VAttrList out_list = VCreateAttrList();
  VAppendAttr(out_list,"image",NULL,VImageRepn,dest);
    
  VSetAttr(VImageAttrList(permutation_dest),"name",NULL,VStringRepn,"PCA SVM Z");
  VAppendAttr(out_list,"image",NULL,VImageRepn,permutation_dest);

  //VHistory(VNumber(program_options),program_options,argv[0],&attribute_list,&out_list);
  VWriteFile(out_file, out_list);

}

