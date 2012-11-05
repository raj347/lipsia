/**
 * 
 * @file vthreshold.cpp 
 * 
 * @author Tilo Buschmann, Johannes Stelzer
 *
 */

// C++ headers
#include <iostream>
#include <iomanip>
#include <sstream>
#include <map>
#include <fstream>
#include <vector>

// C headers
#include <stdlib.h>
#include <stdio.h>

// VIA headers
#include <viaio/Vlib.h>
#include <viaio/VImage.h>
#include <viaio/mu.h>
#include <viaio/option.h>

#include <boost/concept_check.hpp>
#include <boost/multi_array.hpp>
#include <boost/assign.hpp>
#include <boost/progress.hpp>
#include <boost/foreach.hpp>


#ifdef _OPENMP
#include <omp.h>
#endif /*_OPENMP*/

#include "Threshold.h"

using std::cerr;
using std::cout;
using std::endl;
using std::vector;

extern "C" void getLipsiaVersion(char*,size_t);

#ifdef _OPENMP
/**
 * Configure OpenMP
 * 
 * @param[in] nproc number of processing cores to be used
 */
void configure_omp(int nproc) {
  int number_of_cores = omp_get_num_procs();
  if (nproc > 0 && nproc < number_of_cores) 
    number_of_cores = nproc;
  printf("Using %d cores\n",number_of_cores);
  omp_set_num_threads(number_of_cores);
}
#endif /*OPENMP */

/**
 * Input:
 *        - 1 Vista file with n images
 *        - p-value at which to threshold
 *        - 1 or 2 sided
 *
 * Output: 
 *        - 1 Vista file with 1 image of thresholds
 *
 */
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
  FILE *out_file;
  
  VBoolean    is_two_sided    = false;
  VDouble     p               = 0.05;
  VShort      nproc           = 4;

  static VOptionDescRec program_options[] = {
    {"p",   VDoubleRepn,  1, &p,                VOptionalOpt, NULL, "p Threshold" },
    {"j",   VShortRepn,   1, &nproc,            VOptionalOpt, NULL, "number of processors to use, '0' to use all" },
    {"ts",  VBooleanRepn, 1, &is_two_sided,     VOptionalOpt, NULL, "Whether to conduct a two-sided  test"}
  };

  VParseFilterCmd( VNumber (program_options), program_options, argc, argv, &input_file, &out_file);

#ifdef _OPENMP
  // Take care of multiprocessing
  configure_omp(nproc); 
#endif /*OPENMP */
 
  vector<VImage> source_images;

  VAttrList attribute_list  = VReadFile(input_file, NULL);
  fclose(input_file);
  
  if(!attribute_list)
    VError("Error reading image");
  
  VAttrListPosn position;

  for (VFirstAttr(attribute_list, &position); VAttrExists(&position); VNextAttr(&position)) {
    if (VGetAttrRepn(&position) != VImageRepn)
      continue;

    // Extract this image
    VImage image = NULL;
    VGetAttrValue(&position,NULL,VImageRepn,&image);
    source_images.push_back(image);
  }
  cerr << "Number of pictures: " << source_images.size() << endl;

  int pool_size         = source_images.size();
  int number_of_bands   = VImageNBands(source_images.front());
  int number_of_rows    = VImageNRows(source_images.front());
  int number_of_columns = VImageNColumns(source_images.front());

  boost::multi_array<float, 4> pool(boost::extents[number_of_bands][number_of_rows][number_of_columns][pool_size]);

  /*
   */

  cerr << "Converting images to boost format ... ";
  int pool_index = 0;
  BOOST_FOREACH(VImage image, source_images) {
    for(int band(0); band < number_of_bands; band++) {
      for(int row(0); row < number_of_rows; row++) {
        for(int column(0); column < number_of_columns; column++) {
          pool[band][row][column][pool_index] = VGetPixel(image,band,row,column);
        }
      }
    }
    pool_index++;
  }
  cerr << "done" << endl;

  cerr << "Calculating thresholds." << endl;
  Threshold thr;
  boost::multi_array<float, 4> thresholds = thr.calculate(pool, p, is_two_sided);
  cerr << "Done." << endl;
  
  cerr << "Writing out." << endl;
  // Writing it out
  VImage threshold_image_right = VCreateImage(number_of_bands,number_of_rows,number_of_columns,VFloatRepn);
  VFillImage(threshold_image_right, VAllBands, 0);
  VSetAttr(VImageAttrList(threshold_image_right),"name",NULL,VStringRepn,"Threshold Right Side");
  VCopyImageAttrs (source_images.front(), threshold_image_right);

  VImage threshold_image_left;
  if (is_two_sided) {
    threshold_image_left = VCreateImage(number_of_bands,number_of_rows,number_of_columns,VFloatRepn);
    VFillImage(threshold_image_left, VAllBands, 0);
    VSetAttr(VImageAttrList(threshold_image_right),"name",NULL,VStringRepn,"Threshold Left Side");
    VCopyImageAttrs (source_images.front(), threshold_image_left);
  }

  for(int band(0); band < number_of_bands; band++) {
    for(int row(0); row < number_of_rows; row++) {
      for(int column(0); column < number_of_columns; column++) {
        VPixel(threshold_image_right, band, row, column, VFloat) = thresholds[band][row][column][0];

        if (is_two_sided)
          VPixel(threshold_image_left, band, row, column, VFloat) = thresholds[band][row][column][1];

      }
    }
  }
  VAttrList out_list = VCreateAttrList();
  VAppendAttr(out_list,"image",NULL,VImageRepn, threshold_image_right);
  if (is_two_sided)
    VAppendAttr(out_list,"image",NULL,VImageRepn, threshold_image_left);
  
  VHistory(VNumber(program_options),program_options,argv[0],&attribute_list,&out_list);
  VWriteFile(out_file, out_list);
  cerr << "Done." << endl;

}

