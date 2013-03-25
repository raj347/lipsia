/*
** vmerge - merge multiple images into one file
**
** Tilo Buschmann, 2012
*/

// C++ header
#include <iostream>
#include <vector>
#include <boost/foreach.hpp>

// C header
#include <stdio.h>
#include <stdlib.h>

// VIA header
#include <viaio/Vlib.h>
#include <viaio/VImage.h>
#include <viaio/mu.h>
#include <viaio/option.h>

using std::cerr;
using std::cout;
using std::endl;
using std::vector;

extern "C" void getLipsiaVersion(char*,size_t);

int main (int argc,char *argv[]) {
  /**************************
   * Initialise Vista Stuff *
   **************************/

  // Output program name and version
  char version[100];
  getLipsiaVersion(version, sizeof(version));
  cerr << argv[0] << " V" << version << endl;

  /*********************************
   * Parse command line parameters *
   *********************************/

  FILE *output_file;
  VArgVector input_filenames;
  
  static VOptionDescRec program_options[] = {
    {"in", VStringRepn, 0, &input_filenames, VRequiredOpt, NULL, "Input files"}
  };
  VParseFilterCmd(VNumber (program_options),program_options,argc,argv,NULL,&output_file);
  
  /******************************************
   * Read image file and extract image data *
   ******************************************/

  int number_of_vista_files = input_filenames.number;

  VAttrList attribute_list;

  vector<VImage> images;
  
  for(int i(0); i < number_of_vista_files; i++) {
    cerr << "Reading file " << ((VStringConst *) input_filenames.vector)[i] << endl;
    
    /*******************
     * Read vista file *
     *******************/
    VStringConst input_filename = ((VStringConst *) input_filenames.vector)[i];
    FILE *input_file            = VOpenInputFile(input_filename, TRUE);
    attribute_list              = VReadFile(input_file, NULL);
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
      
      VImage current_image        = NULL;
      VGetAttrValue(&position,NULL,VImageRepn,&current_image);
      images.push_back(current_image);
    }
  }

  if (images.size() == 0) {
    VError("Did not find a single image.");
  }
  
  cerr << "Got " << images.size() << " images" << endl;
                                       
  /*******************************
   * Save result into vista file *
   *******************************/

  VAttrList out_list = VCreateAttrList();
  BOOST_FOREACH(VImage current_image, images) {
    VAppendAttr(out_list,"image",NULL,VImageRepn,current_image);
  }
  VWriteFile(output_file, out_list);
}

