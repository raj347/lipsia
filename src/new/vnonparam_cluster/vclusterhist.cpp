/**
 * 
 * @file vclusterhist.cpp 
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

extern "C" VImage VLabelImage3d(VImage, VImage, int, VRepnKind, int *);
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
  cerr << "Using " << number_of_cores << " core(s)" << endl;
  omp_set_num_threads(number_of_cores);
}
#endif /*OPENMP */

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
  
  VShort      nproc             = 4;
  VString     t_input_filename  = NULL;
  VString     hist_filename     = NULL;

  static VOptionDescRec program_options[] = {
    {"tin",   VStringRepn,  1, &t_input_filename, VRequiredOpt, NULL, "Threshold input file" },
    {"hist",  VStringRepn,  1, &hist_filename,    VRequiredOpt, NULL, "Histogramm output file" },
    {"j",     VShortRepn,   1, &nproc,            VOptionalOpt, NULL, "number of processors to use, '0' to use all" }
  };

  VParseFilterCmd( VNumber (program_options), program_options, argc, argv, &input_file, NULL);

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

  FILE *t_input_file = fopen(t_input_filename,"r");
  VAttrList t_attribute_list  = VReadFile(t_input_file, NULL);
  fclose(t_input_file);

  if(!t_attribute_list)
    VError("Error reading threshold image");

  VAttrListPosn t_position;
  
  VImage t_images[2];

  int nr_images = 0;
  for (VFirstAttr(t_attribute_list, &t_position); VAttrExists(&t_position) && (nr_images < 2); VNextAttr(&t_position)) {
    if (VGetAttrRepn(&t_position) != VImageRepn)
      continue;

    // Extract this image
    VImage image = NULL;
    VGetAttrValue(&t_position,NULL,VImageRepn,&image);
    t_images[nr_images] = image;
    nr_images++;
  }
  
  if ((nr_images == 0) || (nr_images > 2)) {
    cerr << "Does not look like a threshold image" << endl;
    exit(-1);
  }

  int number_of_bands   = VImageNBands(source_images.front());
  int number_of_rows    = VImageNRows(source_images.front());
  int number_of_columns = VImageNColumns(source_images.front());

  cerr << "Writing out." << endl;

  /*
  BOOST_FOREACH(VImage old_image, source_images) {
    int band    = 14;
    int row     = 18;
    int column  = 111;

    cerr << band << "/" << row << "/" << column << endl;
    cerr << "Pixel: " << VGetPixel(old_image,band, row, column) << "  " << VGetPixel(t_images[0],band, row, column) << "  "<< VGetPixel(t_images[1],band, row, column) << endl;
  }
  exit(-1);
  */

  VImage label_image = VCreateImage(number_of_bands, number_of_rows , number_of_columns, VShortRepn);

  vector<double>  cluster_sizes;

  int max_cluster_size = 0;

  BOOST_FOREACH(VImage old_image, source_images) {
    VImage new_image = VCreateImage(number_of_bands,number_of_rows,number_of_columns, VBitRepn);
    VFillImage(new_image, VAllBands, 0);
    VCopyImageAttrs (old_image, new_image);
    VSetAttr(VImageAttrList(new_image),"name",NULL,VStringRepn,"Binarization");

    for(int band(0); band < number_of_bands; band++) {
      for(int row(0); row < number_of_rows; row++) {
        for(int column(0); column < number_of_columns; column++) {
          int presence = 0;

          if (VGetPixel(old_image, band, row, column) > VGetPixel(t_images[0],band, row, column))
            presence = 1;
          if ((nr_images == 2) && VGetPixel(old_image, band, row, column) < VGetPixel(t_images[1],band, row, column))
            presence = 1;

          VPixel(new_image, band, row, column, VBit) = presence;

        }
      }
    }
    // Let's count clusters in this picture
    int number_of_labels;
    VFillImage(label_image,VAllBands,0);
    //label_image = VLabelImage3d(new_image, label_image, 26, VShortRepn, &number_of_labels);
    label_image = VLabelImage3d(new_image, label_image, 6, VShortRepn, &number_of_labels);

    vector<double>  cluster_size(number_of_labels);       // [0..number_of_labels-1]
    vector<int>     cluster_dimension(number_of_labels);  // [0..number_of_labels-1]

    for (int label = 0; label < number_of_labels; label++)  {
      cluster_size[label]       = 0.0;
      cluster_dimension[label]  = 0;
    }

    for(int band(0); band < number_of_bands; band++) {
      for(int row(0); row < number_of_rows; row++) {
        for(int column(0); column < number_of_columns; column++) {
          short label = VPixel(label_image, band, row, column, VShort);

          if ((label > number_of_labels) || (label < 0)) {
            cerr << "Got unexpected label: " << label << endl;
            exit(-1);
          }
          if (label >= 1) {
            cluster_dimension[label-1]++;
            if (VGetPixel(old_image, band, row, column) > VGetPixel(t_images[0], band, row, column)) {
              //cluster_size[label-1] += (VGetPixel(old_image, band, row, column) / VGetPixel(t_images[0],band, row, column) -1);
              cluster_size[label-1] += 1;
            } else if ((nr_images == 2) && VGetPixel(old_image, band, row, column) < VGetPixel(t_images[1],band, row, column)) {
              //cluster_size[label-1] += (VGetPixel(old_image, band, row, column) / VGetPixel(t_images[1],band, row, column) -1);
              cluster_size[label-1] += 1;
            }
          }
        }
      }
    }
    //cerr << "Cluster Nr 1 dimension=" << cluster_dimension[0] << " size=" << cluster_size[0] << endl;

    for (int label = 0; label < number_of_labels; label++) {
      if (cluster_dimension[label] > 1) {
        cluster_sizes.push_back(cluster_size[label]);
        if (cluster_size[label] > max_cluster_size)
          max_cluster_size = cluster_size[label];
      }
    }
  }

  double N = cluster_sizes.size();
  /*
  BOOST_FOREACH(double this_cluster_size, cluster_sizes) {
    N += this_cluster_size;
  }
  */

  vector<double> cluster_size_p;

  sort(cluster_sizes.begin(),cluster_sizes.end(),std::greater<double>());

  double cum_sum = 0;
  // FIXME totally unnecessary
  BOOST_FOREACH(double this_size, cluster_sizes) {
    cum_sum += 1.0;
    cluster_size_p.push_back(cum_sum / N);
  }

  FILE *hist_file = fopen(hist_filename,"w");  
  for (unsigned int i = 0; i < cluster_sizes.size(); i++) {
    fprintf(hist_file,"%e\t%e\n",cluster_sizes[i],cluster_size_p[i]);
  }
  fclose(hist_file);

  cerr << "Done." << endl;
}

