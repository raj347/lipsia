/*
** SVM - Support Vector Machines
**
** Tilo Buschmann, 2012
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

// VIA header
#include <viaio/Vlib.h>
#include <viaio/VImage.h>
#include <viaio/mu.h>
#include <viaio/option.h>

// Class header
#include "MriSvm.h"
#include "SearchLight.h"

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

int parseSvmType(VString svm_type) {
  if (NULL == svm_type)
    return DEFAULT_MRISVM_SVM_TYPE;

  static map<string, MriSvm::SVM_TYPE> svm_type_map = map_list_of("C_SVC",      MriSvm::C_SVC)
                                                                 ("NU_SVC",     MriSvm::NU_SVC)
                                                                 ("ONE_CLASS",  MriSvm::ONE_CLASS)
                                                                 ("EPSILON_SVR",MriSvm::EPSILON_SVR)
                                                                 ("NU_SVR",     MriSvm::NU_SVR);

  map<string, MriSvm::SVM_TYPE>::iterator it;
  
  it = svm_type_map.find(svm_type);

  if (it == svm_type_map.end()) {
    return DEFAULT_MRISVM_SVM_TYPE;
  }

  return it->second;

}

int parseSvmKernelType(VString svm_kernel_type) {
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
    return DEFAULT_MRISVM_KERNEL_TYPE;
  }

  return it->second;

}


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
  VDouble           svm_C           = DEFAULT_MRISVM_C;
  VDouble           svm_gamma       = DEFAULT_MRISVM_GAMMA;
  VString           svm_type        = NULL;
  VString           svm_kernel_type = NULL;

  static VOptionDescRec program_options[] = {
    {"in",          VStringRepn, 0, &input_filenames, VRequiredOpt, NULL, "Input files" },
    {"svm_type",    VStringRepn, 1, &svm_type,        VOptionalOpt, NULL, "SVM Type" },
    {"kernel_type", VStringRepn, 1, &svm_kernel_type, VOptionalOpt, NULL, "Kernel Type" },
    {"C",           VDoubleRepn, 1, &svm_C,           VOptionalOpt, NULL, "SVM C parameter" },
    {"gamma",       VDoubleRepn, 1, &svm_gamma,       VOptionalOpt, NULL, "SVM gamma parameter" }
  };
  VParseFilterCmd(VNumber (program_options),program_options,argc,argv,NULL,NULL);

  // Translate SVM parameter strings to enums
  int svm_type_parsed         = parseSvmType(svm_type);
  int svm_kernel_type_parsed  = parseSvmKernelType(svm_kernel_type);

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

  sample_features_array_type sample_features(boost::extents[number_of_images][number_of_features]);

  vector <int> classes(number_of_images);

  cerr << "Converting Data into MriSvm Format" << endl;
  boost::progress_display convert_progress(number_of_images);
  for(int sample_index(0); sample_index < number_of_images; sample_index++) {
    ++convert_progress;
    //cerr << "Converting data from image " << sample_index << endl;

    long image_class = DEFAULT_VSVM_IMAGE_CLASS;
    if(VGetAttr(VImageAttrList(source_images[sample_index]), "class", NULL, VLongRepn, &image_class) != VAttrFound) {
      cerr << "Image does not have class attribute. Using default value (" << DEFAULT_VSVM_IMAGE_CLASS << ")" << endl;
    }
    classes[sample_index] = image_class;

    int feature_index(0);
    for(int band(0); band < VImageNBands(source_images[sample_index]); band++) {
      for(int row(0); row < VImageNRows(source_images[sample_index]); row++) {
        for(int column(0); column < VImageNColumns(source_images[sample_index]); column++) {
          sample_features[sample_index][feature_index] = VGetPixel(source_images[sample_index],band,row,column);
          feature_index++;
        }
      }
    }
  }

  /***************
   * Conduct SVM *
   ***************/

  // Initialise SVM 
  MriSvm mrisvm(
                sample_features,
                classes,
                number_of_images,
                number_of_features
               );

  mrisvm.set_svm_type(svm_type_parsed);
  mrisvm.set_svm_kernel_type(svm_kernel_type_parsed);
  // Scale features
  mrisvm.scale();

  // Validate
  cerr << "Cross validating" << endl;
  double validity = mrisvm.cross_validate(2);
  cerr << "Cross Validation Accuracy = " << 100.0*validity << endl; 

  // Export
  //mrisvm.export_table(std::string("raw_data.tab"));
}

