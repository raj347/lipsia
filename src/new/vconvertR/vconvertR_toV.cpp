/**
 * @file vconvertR_toV.cpp
 * 
 * Convert Data in table format to Vista
 *
 * @author Tilo Buschmann
 */

// C++ header
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

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

#define DEFAULT_VSVM_IMAGE_CLASS 0

using std::cerr;
using std::cout;
using std::endl;
using std::vector;
using std::ofstream;

extern "C" void getLipsiaVersion(char*,size_t);

int main (int argc,char *argv[]) {
  /**************************
   * Initialise Vista Stuff *
   **************************/

  // Output program name and version
  char version[100];
  getLipsiaVersion(version, sizeof(version));
  cerr << argv[0] << " V" << version << endl;

  // Parse command line parameters
  FILE *output_file;
  VString original_filename;
  VString input_filename;

  static VOptionDescRec program_options[] = {
    {"in",VStringRepn,1,&input_filename,VRequiredOpt, NULL, "Filename if tab delimited file" },
    {"original",VStringRepn,1,&original_filename,VRequiredOpt, NULL, "One of the original samples (to copy attributes)" },
  };

  VParseFilterCmd(VNumber (program_options),program_options,argc,argv,NULL,&output_file);

  // Read original file
  FILE *original_file = VOpenInputFile(original_filename, TRUE);
  VAttrList attribute_list      = VReadFile(original_file, NULL);
  fclose(original_file);
  if(!attribute_list)
    VError("Error reading original image");

  VImage original_image = NULL;
  VAttrListPosn position;

  int number_of_features_per_voxel  = 0;
  for (VFirstAttr(attribute_list, &position); VAttrExists(&position); VNextAttr(&position)) {
    // Skip attribute if it is not an image
    if (VGetAttrRepn(&position) != VImageRepn)
      continue;
    VGetAttrValue(&position,NULL,VImageRepn,&original_image);
    number_of_features_per_voxel++;
  }

  if (number_of_features_per_voxel == 0) {
      VError("No original image found");
  }
  
  int number_of_bands   = VImageNBands(original_image);
  int number_of_rows    = VImageNRows(original_image);
  int number_of_columns = VImageNColumns(original_image);

  /*
   * New Images (one per feature)
   */

  VImage *destinations = new VImage[number_of_features_per_voxel];

  for (int feature(0); feature < number_of_features_per_voxel; feature++) {
    destinations[feature] = VCreateImage(number_of_bands,number_of_rows,number_of_columns,VFloatRepn);
    VImage dest = destinations[feature];
    VFillImage(dest,VAllBands,0);
    VCopyImageAttrs(original_image, dest);
    VSetAttr(VImageAttrList(dest),"name",NULL,VStringRepn,"PCA+SVM");
  }

  // Here be magic
  std::ifstream input_file(input_filename);
  std::string   line;
  int band,row,column,feature;
  double value;

  while(getline(input_file,line)) { 
      std::stringstream   linestream(line);
      linestream >> band >> row >> column >> feature >> value;
      if (band >= 0 && band < number_of_bands && row >= 0 && row < number_of_rows && column >= 0 && column < number_of_columns && feature >= 0 && feature < number_of_features_per_voxel)
        VPixel(destinations[feature],band,row,column,VFloat) = value;
  }

  VAttrList out_list = VCreateAttrList();
  for (int feature(0); feature < number_of_features_per_voxel; feature++) {
    VAppendAttr(out_list,"image",NULL,VImageRepn,destinations[feature]);
  }

  VHistory(VNumber(program_options),program_options,argv[0],&attribute_list,&out_list);
  VWriteFile(output_file,out_list);

  delete[] destinations;  
}

