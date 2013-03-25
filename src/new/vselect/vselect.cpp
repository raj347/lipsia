/*
** vselect - extract an image from a vista file
**
** Tilo Buschmann, 2012
*/

// C++ header
#include <iostream>

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
  VShort  image_position = 0;

  FILE *output_file,*input_file;

  static VOptionDescRec program_options[] = {
    {"position", VShortRepn, 1, &image_position, VRequiredOpt, NULL, "Position of selected Image" }
  };
  VParseFilterCmd(VNumber (program_options),program_options,argc,argv,&input_file,&output_file);
  
  /******************************************
   * Read image file and extract image data *
   ******************************************/

  VAttrList attribute_list  = VReadFile(input_file, NULL);
  fclose(input_file);
  
  if(!attribute_list)
    VError("Error reading image");
  
  VImage selected_image = NULL;
  VAttrListPosn position;
  int image_index = 0;
  
  for (VFirstAttr(attribute_list, &position); VAttrExists(&position); VNextAttr(&position)) {
    if (VGetAttrRepn(&position) != VImageRepn)
      continue;
    image_index++;
    
    if(image_index == image_position) {
      VGetAttrValue(&position,NULL,VImageRepn,&selected_image);
      VString name;
      VGetAttr(VImageAttrList(selected_image), "name", NULL, VStringRepn, &name);
      cerr << "  Name: " << name << endl;
      break;
    }
  }
  
  if (selected_image == NULL) 
    VError("No input image found");
                                       
  /*******************************
   * Save result into vista file *
   *******************************/

  VAttrList out_list = VCreateAttrList();
  VAppendAttr(out_list,"image",NULL,VImageRepn,selected_image);
  VHistory(VNumber(program_options),program_options,argv[0],&attribute_list,&out_list);
  VWriteFile(output_file, out_list);
}

