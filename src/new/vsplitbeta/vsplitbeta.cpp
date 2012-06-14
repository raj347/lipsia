/*
* vsplitbeta - split betas from a single file
*
* Tilo Buschmann, 2012
*
*/

// C++ header
#include <iostream>
#include <iomanip>
#include <sstream>
#include <map>
#include <fstream>

// C header
#include <stdlib.h>
#include <stdio.h>

// VIA header
#include <viaio/Vlib.h>
#include <viaio/VImage.h>
#include <viaio/mu.h>
#include <viaio/option.h>
#include <boost/concept_check.hpp>
#include <boost/concept_check.hpp>

using std::cerr;
using std::cout;
using std::endl;

extern "C" void getLipsiaVersion(char*,size_t);

// Map from beta to class
typedef std::map<int,int> mapT; 
  
// My own version of VReadHistory that does not suck as much
VAttrList _VReadHistory_(VAttrList *list) { 
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

void read_parse_conversion_file(VString conversion_file,mapT &conversion_map) {
  std::ifstream file(conversion_file);
  std::string   line;

  while(std::getline(file, line)) {
    std::stringstream   linestream(line);
    int                 val1;
    int                 val2;

    linestream >> val1 >> val2;
    conversion_map[val1] = val2;
  }
  file.close();
}

bool is_beta_image(VImage &image) {
    VString name;
    VGetAttr(VImageAttrList(image), "name", NULL, VStringRepn, &name);
    return(0 == strcmp(name,"BETA"));
}

bool converte_beta_to_class(VImage &image, mapT &conversion_map, int &beta, int &conversion_value) {
  VShort beta_value;
  VGetAttr(VImageAttrList(image), "beta", NULL, VShortRepn, &beta_value);
  mapT::iterator it = conversion_map.find(beta_value);
  if (it == conversion_map.end())
    return false;
  
  conversion_value  = it->second;
  beta              = beta_value;
  
  return true;
}

std::string construct_filename(VString output_base,int beta,int svm_class) {
  std::stringstream filename_stream;
  filename_stream << output_base << "_" << std::setw(4) << std::setfill('0') << beta << "_" << svm_class << ".v";
  return(filename_stream.str());
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
  VString output_base = NULL; // Extrated images will be stored in files with the name outputbase_betanr_class.v
  VString conversion_file = NULL; // The file that contains the conversion beta->class

  static VOptionDescRec program_options[] = {
    {"base", VStringRepn, 1, &output_base, VRequiredOpt, NULL, "basename of extracted images" },
    {"conversion", VStringRepn, 1, &conversion_file, VRequiredOpt, NULL, "name of conversion file" }
  };
  VParseFilterCmd(VNumber (program_options),program_options,argc,argv,&input_file,NULL);
 
  cerr << "Base filename: " << output_base << endl; 

  mapT conversion_map;
  read_parse_conversion_file(conversion_file,conversion_map);

  VAttrList attribute_list  = VReadFile(input_file, NULL);
  fclose(input_file);
  
  if(!attribute_list)
    VError("Error reading image");
  
  VAttrListPosn position;
  bool first_history = true;

  for (VFirstAttr(attribute_list, &position); VAttrExists(&position); VNextAttr(&position)) {
    if (VGetAttrRepn(&position) != VImageRepn)
      continue;

    // Extract this image
    VImage image = NULL;
    VGetAttrValue(&position,NULL,VImageRepn,&image);
    
    // Retrieve name of image and do nothing if image is not a beta map
    if (!is_beta_image(image))
      continue;

    // Find this beta in the beta->class conversion map, do nothing if not found
    int beta,svm_class;
    if (!converte_beta_to_class(image,conversion_map,beta,svm_class))
      continue;
    
    std::string filename = construct_filename(output_base,beta,svm_class);
    
    VSetAttr(VImageAttrList(image),"class",NULL,VShortRepn,svm_class);
    
    /*******************************
      * Save result into vista file *
      *******************************/
    FILE *output_file = fopen(filename.c_str(), "w");
    if (NULL != output_file) {
      
      VAttrList out_list = VCreateAttrList();
      
      VAppendAttr(out_list,"image",NULL,VImageRepn,image);
      
      // This aweful VHIstory function has a side effect and does indeed change the original attribute list!
      if (first_history) {
        VHistory(VNumber(program_options),program_options,argv[0],&attribute_list,&out_list);
        first_history = false;
      } else {
        VAttrList history_list = _VReadHistory_(&attribute_list);
        VPrependAttr(out_list,"history",NULL,VAttrListRepn,history_list);
      }
      VWriteFile(output_file, out_list);
      
      fclose(output_file);
    } else {
      cerr << "Some error occurred" << endl;
    }
  }
}

