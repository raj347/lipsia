/**
 * 
 * @file vmontecarlo.cpp 
 * 
 * @author Tilo Buschmann, Johannes Stelzer, Enrico Reimer
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
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/variate_generator.hpp>


#ifdef _OPENMP
#include <omp.h>
#endif /*_OPENMP*/

using std::cerr;
using std::cout;
using std::endl;
using std::vector;

//extern "C" void getLipsiaVersion(char*,size_t);

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

int main (int argc,char *argv[]) {
  /**************************
   * Initialise Vista Stuff *
   **************************/

  // Output program name and version
  char version[100];
  //getLipsiaVersion(version, sizeof(version));
  cerr << argv[0] << " V" << version << endl;

  /*********************************
   * Parse command line parameters *
   *********************************/
  VArgVector  input_filenames;

  FILE *out_file;
  
  VShort      nproc           = 4;
  VShort      nsteps          = 1000;

  static VOptionDescRec program_options[] = {
    {"in",    VStringRepn,  0, &input_filenames,  VRequiredOpt, NULL, "Input files" },
    {"steps", VShortRepn,   1, &nsteps,           VOptionalOpt, NULL, "number of sampled volumes" },
    {"j",     VShortRepn,   1, &nproc,            VOptionalOpt, NULL, "number of processors to use, '0' to use all" }
  };

  VParseFilterCmd( VNumber (program_options), program_options, argc, argv, NULL, &out_file);

#ifdef _OPENMP
  // Take care of multiprocessing
  configure_omp(nproc); 
#endif /*OPENMP */

  size_t nsubjects  = input_filenames.number;
  size_t nperms     = 0;

  cerr << "Reading Image Files ... " << endl;

  std::vector<VImage> subject_images[nsubjects];

  VAttrList attribute_list;

  for (int subject_idx(0); subject_idx < nsubjects; subject_idx++) {
      subject_images[subject_idx] = std::vector<VImage>();

      VStringConst input_filename = ((VStringConst *) input_filenames.vector)[subject_idx];
      std::cerr << input_filename << std::endl;

      FILE *input_file          = VOpenInputFile(input_filename, TRUE);
      attribute_list  = VReadFile(input_file, NULL);
      fclose(input_file);

      if(!attribute_list)
        VError("Error reading image");

      VAttrListPosn position;
      for (VFirstAttr(attribute_list, &position); VAttrExists(&position); VNextAttr(&position)) {
        if (VGetAttrRepn(&position) != VImageRepn)
          continue;
        VImage image;
        VGetAttrValue(&position,NULL,VImageRepn,&image);
        subject_images[subject_idx].push_back(image);

        // TODO: Check consistency of image dimensions
      }
      if (nperms == 0)
        nperms = subject_images[subject_idx].size();
      else if (nperms != subject_images[subject_idx].size())
        VError("Number of permutations not consistent");
  }
  std::cerr << "Number of permutations per subject: " << nperms << std::endl;

  VImage template_image = subject_images[0].front();

  int number_of_bands   = VImageNBands(template_image);
  int number_of_rows    = VImageNRows(template_image);
  int number_of_columns = VImageNColumns(template_image);
	
  boost::mt19937 gen;
	boost::uniform_int<> dist(0, nperms-1);
	boost::variate_generator<boost::mt19937&, boost::uniform_int<> > die(gen, dist);

  VAttrList out_list = VCreateAttrList();

  boost::progress_display writing_progress(nsteps);

#pragma omp parallel for firstprivate(die)
  for(int step = 0; step < nsteps; step++) {

    VImage dest = VCreateImage(number_of_bands,number_of_rows,number_of_columns,VFloatRepn);
    VFillImage(dest,VAllBands,0);
    VCopyImageAttrs(template_image, dest);

    for (int subject_idx = 0; subject_idx < nsubjects; subject_idx++) {
      int perm_idx = die();
      VImage perm_image = subject_images[subject_idx].at(perm_idx);

      for (int band = 0; band < number_of_bands; band++) {
        for (int row = 0; row < number_of_rows; row++) {
          for (int column = 0; column < number_of_columns; column++) {
            VPixel(dest,band,row,column,VFloat) += VPixel(perm_image,band,row,column,VFloat);
          }
        }
      }
    }
    for (int band = 0; band < number_of_bands; band++) {
      for (int row = 0; row < number_of_rows; row++) {
        for (int column = 0; column < number_of_columns; column++) {
          VPixel(dest,band,row,column,VFloat) /= (float) nsubjects;
        }
      }
    }

#pragma omp critical
    VAppendAttr(out_list,"image",NULL,VImageRepn,dest);

#pragma omp critical
    ++writing_progress;
  }

  cerr << "Saving to disk ...";
  VWriteFile(out_file, out_list);
  fclose(out_file);
  cerr << "done." << endl;


}

