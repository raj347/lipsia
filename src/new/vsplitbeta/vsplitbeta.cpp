/*
** vsplitbeta - split betas from a single file
**
** Tilo Buschmann, 2012
**
** Screw it, I am using C++ in this file
*/

// C++ header
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cstdio>
#include <map>
#include <fstream>

// C header
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

// My own version of VReadHistory that does not suck as much
VAttrList VReadHistory(VAttrList *list) { 
  VAttrListPosn posn;
  VAttrList history_list=NULL;
  char history[]="history";

  for (VLastAttr((*list),&posn);VAttrExists(&posn);VPrevAttr(&posn)) {
    if (strncmp(VGetAttrName(&posn), history, strlen(history)) != 0 )
      continue;

    if (VGetAttrRepn(&posn) == VAttrListRepn ) {
      VGetAttrValue(&posn, NULL, VAttrListRepn, &history_list);
      break;
    }
  }

  return history_list;
}


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
  FILE *input_file;

  VString output_base = NULL;
  VString conversion_file = NULL;

  static VOptionDescRec program_options[] = {
    {"base", VStringRepn, 1, &output_base, VRequiredOpt, NULL, "basename of extracted images" },
    {"conversion", VStringRepn, 1, &conversion_file, VRequiredOpt, NULL, "name of conversion file" }
  };
  VParseFilterCmd(VNumber (program_options),program_options,argc,argv,&input_file,NULL);
 
  cerr << "Base filename: " << output_base << endl; 

  /************************
   * Read Conversion file *
   ************************/ 

  // From beta to class
  typedef std::map<int,int> mapT; 
  mapT conversion_map;

  std::ifstream file(conversion_file);
  std::string   line;

  while(std::getline(file, line)) {
    std::stringstream   linestream(line);
    int                 val1;
    int                 val2;

    linestream >> val1 >> val2;

    conversion_map[val1] = val2;
  }

  /******************************************
   * Read image file and extract image data *
   ******************************************/

  VAttrList attribute_list  = VReadFile(input_file, NULL);
  fclose(input_file);
  
  if(!attribute_list)
    VError("Error reading image");
  
  VAttrListPosn position;
  bool first_history = true;

  for (VFirstAttr(attribute_list, &position); VAttrExists(&position); VNextAttr(&position)) {
    if (VGetAttrRepn(&position) != VImageRepn)
      continue;


    VImage selected_image = NULL;

    VGetAttrValue(&position,NULL,VImageRepn,&selected_image);
    VString name;
    VGetAttr(VImageAttrList(selected_image), "name", NULL, VStringRepn, &name);

    if (0 == strcmp(name,"BETA")) {
        
      // Find this beta in the conversion map
      VShort beta_value;
      VGetAttr(VImageAttrList(selected_image), "beta", NULL, VShortRepn, &beta_value);

      mapT::iterator it = conversion_map.find(beta_value);

      if (it != conversion_map.end()) {
        int conversion_value = it->second;
      
        std::stringstream filename_stream;
        filename_stream << output_base << "_" << std::setw(4) << std::setfill('0') << beta_value << ".v";
        std::string filename = filename_stream.str();
        cerr << filename << endl;
     
        VSetAttr(VImageAttrList(selected_image),"class",NULL,VShortRepn,conversion_value);

        /*******************************
         * Save result into vista file *
         *******************************/

        FILE *output_file;
        output_file = fopen(filename.c_str(), "w");
        if (NULL != output_file) {

          VAttrList out_list = VCreateAttrList();

          VAppendAttr(out_list,"image",NULL,VImageRepn,selected_image);

          // This aweful VHIstory function has a side effect and does indeed change the original attribute list!
          if (first_history) {
            VHistory(VNumber(program_options),program_options,argv[0],&attribute_list,&out_list);
            first_history = false;
          } else {
            VAttrList history_list = VReadHistory(&attribute_list);
            VPrependAttr(out_list,"history",NULL,VAttrListRepn,history_list);
          }
          VWriteFile(output_file, out_list);

          fclose(output_file);
        } else {
          cerr << "Some error occurred" << endl;
        }

      }

    }
  }
}

