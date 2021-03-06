/**
 * @file vconvertR.cpp
 * 
 * Convert Data in our format to something readable by R
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
#include <viaio/file.h>

#define DEFAULT_VSVM_IMAGE_CLASS 0

using std::cerr;
using std::cout;
using std::endl;
using std::vector;
using std::ofstream;

extern "C" void getLipsiaVersion(char*,size_t);

bool are_coordinates_valid(int band,int row, int column, int number_of_bands, int number_of_rows, int number_of_columns) {
  return ((band >= 0) && (band < number_of_bands) && (column >= 0) && (column < number_of_columns) && (row >= 0) && (row < number_of_rows));
}

bool is_voxel_or_surrounding_filled(VImage this_image,int band,int row,int column,int number_of_bands, int number_of_rows, int number_of_columns) {
  if (VGetPixel(this_image,band,row,column) != 0.0)
    return true;
  else
    return false;

  int surrounding = 0;
  for (int rel_band_index = -1;rel_band_index <= 1; rel_band_index++) {
    for (int rel_row_index = -1;rel_row_index <= 1; rel_row_index++) {
      for (int rel_column_index = -1;rel_column_index <= 1; rel_column_index++) {
        int this_band = band + rel_band_index;
        int this_row = row + rel_row_index;
        int this_column = column + rel_column_index;

        if (are_coordinates_valid(this_band,this_row,this_column,number_of_bands,number_of_rows,number_of_columns) && VGetPixel(this_image,this_band,this_row,this_column) != 0.0)
          surrounding++;
      }
    }
  }
  return (surrounding > 14);
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
  VString output_filename = NULL;
  VString mask_filename   = NULL;
  VBoolean  filter_empty = false;
  VShort  image_position = 0;

  static VOptionDescRec program_options[] = {
    {"in",        VStringRepn,  0, &input_filenames,  VRequiredOpt, NULL, "Input files" },
    {"out",       VStringRepn,  1, &output_filename,  VRequiredOpt, NULL, "Output file" },
    {"mask",      VStringRepn,  1, &mask_filename,     VOptionalOpt, NULL, "Mask file" },
    {"position",  VShortRepn,   1, &image_position,   VRequiredOpt, NULL, "Position of selected Image (0=all)" },
    {"filter",    VBooleanRepn, 1, &filter_empty,     VOptionalOpt, NULL, "Whether to filter out empty voxels"}
  };
  VParseFilterCmd(VNumber (program_options),program_options,argc,argv,NULL,NULL);

  VImage mask = NULL;

  if (mask_filename != NULL) {
    FILE *fp        = VOpenInputFile(mask_filename, TRUE);
    VAttrList list  = VReadFile(fp, NULL);
    fclose(fp);

    if(!list)
      VError("Error reading mask image");

    VAttrListPosn posn;

    for (VFirstAttr(list, &posn); VAttrExists(&posn); VNextAttr(&posn)) {

      if(VGetAttrRepn(& posn) != VImageRepn)
        continue;

      VGetAttrValue(&posn, NULL, VImageRepn, &mask);

      if(VPixelRepn(mask) != VBitRepn)
        VError(" mask image must be of type bit");

      break;
    }
  }


  /*******************************************
   * Read image files and extract image data *
   *******************************************/

  int number_of_samples = input_filenames.number;
  vector<VImage> *source_images = new vector<VImage>[number_of_samples]; // clang++ demands dynamical allocation of arrays of non-POD

  long int  number_of_features = 0;
  int       number_of_features_per_voxel = 0;
  VAttrList attribute_list;
  
  cerr << "Reading Image Files" << endl;

  boost::progress_display file_progress(number_of_samples);
  for(int i(0); i < number_of_samples; i++) {
    ++file_progress;

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
    int this_number_of_features_per_voxel = 0;
    int image_index = 0;

    for (VFirstAttr(attribute_list, &position); VAttrExists(&position); VNextAttr(&position)) {
      if (VGetAttrRepn(&position) != VImageRepn)
        continue;
      image_index++;

      if (0 == image_position || image_index == image_position) {
        VImage image;
        VGetAttrValue(&position,NULL,VImageRepn,&image);

        // Get name (i.e. what type of image this is)
        //VString name;
        //VGetAttr(VImageAttrList(image), "name", NULL, VStringRepn, &name);
        //cerr << "  Name: " << name << endl;

        source_images[i].push_back(image);
        this_number_of_features_per_voxel++;

        // Get number of features (i.e. bands * rows * columns)
        int this_number_of_features = VImageNPixels(image);
        if (0 == number_of_features) {
          number_of_features = this_number_of_features;
        } else if (number_of_features != this_number_of_features) {
          VError("Error: Number of features differs from number of features in previous pictures.");
        }
      }
    }
    
    if (this_number_of_features_per_voxel == 0) 
      VError("No input image found");
    
    if (number_of_features_per_voxel == 0) {
      number_of_features_per_voxel = this_number_of_features_per_voxel;
    } else if (number_of_features_per_voxel != this_number_of_features_per_voxel) {
      VError("This file has a different number of images than previous files.");
    }
  }

  /*****************************
   * Convert to usable format  *
   *****************************/

  ofstream outputr;
  outputr.open(output_filename);


  int number_of_bands   = VImageNBands(source_images[0].front());
  int number_of_rows    = VImageNRows(source_images[0].front());
  int number_of_columns = VImageNColumns(source_images[0].front());


  // Detect empty voxels
  bool voxel_is_empty[number_of_bands][number_of_rows][number_of_columns];
  if (filter_empty) {
    for(int band(0); band < number_of_bands; band++) {
      for(int row(0); row < number_of_rows; row++) {
        for(int column(0); column < number_of_columns; column++) {
          voxel_is_empty[band][row][column] = true;
          for(int sample_index(0); sample_index < number_of_samples; sample_index++) {
            for(int feature(0); feature < number_of_features_per_voxel; feature++) {
              VImage this_image = source_images[sample_index][feature];
              if (is_voxel_or_surrounding_filled(this_image,band,row,column, number_of_bands, number_of_rows, number_of_columns)) {
                voxel_is_empty[band][row][column] = false;
              }
            }
          }
        }
      }
    }
  }
 
  // Header
  for(int sample_index(0); sample_index < number_of_samples; sample_index++) {
    outputr << "\t" << ((VStringConst *) input_filenames.vector)[sample_index];
  }
  outputr << endl;
 
//  // Classes
//  outputr << "Class";
//  for(int sample_index(0); sample_index < number_of_samples; sample_index++) {
//    double image_class = DEFAULT_VSVM_IMAGE_CLASS;
//
//    if(VGetAttr(VImageAttrList(source_images[sample_index].front()), "class", NULL, VDoubleRepn, &image_class) != VAttrFound) {
//      cerr << "Image does not have class attribute. Using default value (" << DEFAULT_VSVM_IMAGE_CLASS << ")" << endl;
//    }
//    outputr << "\t" << image_class;
//  }
//  outputr << endl;

  // Data
  if (mask != NULL) {
    cerr << "Will filter based on mask" << endl;
  }

  boost::progress_display convert_progress(number_of_bands * number_of_rows * number_of_columns);
  for(int band(0); band < number_of_bands; band++) {
    for(int row(0); row < number_of_rows; row++) {
      for(int column(0); column < number_of_columns; column++) {
        bool filter_voxel = filter_empty && ((mask == NULL && voxel_is_empty[band][row][column]) || (mask != NULL && VGetPixel(mask,band,row,column) == 0));

        if (!filter_voxel) {
          for (int feature(0); feature < number_of_features_per_voxel; feature++) {
            outputr << "\"" << band << " " << row << " " << column << " " << feature << "\"";
            for (int sample_index(0); sample_index < number_of_samples; sample_index++) {
              VImage this_image = source_images[sample_index][feature];
              outputr << "\t" << VGetPixel(this_image,band,row,column);
            }
            outputr << endl;
          }
        }
        ++convert_progress;
      }
    }
  }
  outputr.close();
  delete[] source_images;
}

