/**
 * 
 * @file vclusterassess.cpp 
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

using std::cerr;
using std::cout;
using std::endl;
using std::vector;

extern VImage VLabelImage3d(VImage, VImage, int, VRepnKind, int *);
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

std::vector<double> read_hist_file(VString hist_file) {
  std::map<int,double> size_p_map;

  std::ifstream file(hist_file);
  std::string   line;

  int max_cluster_size = 0;
  while(std::getline(file, line)) {
    std::stringstream   linestream(line);
    int                 val1;
    double              val2;

    linestream >> val1 >> val2;
    size_p_map[val1] = val2;

    if (val1 > max_cluster_size)
      max_cluster_size = val1;
  }
  file.close();

  // Sanity check
  if (max_cluster_size > 1e6) {
    cerr << "Cluster size seems to be too large" << endl;
    exit(-1);
  }

  // convert to regular vector
  std::vector<double> size_p_vec(max_cluster_size + 1,0.0);

  for(std::map<int,double>::iterator iter = size_p_map.begin(); iter != size_p_map.end(); ++iter)
  {
    int     size  =  iter->first;
    double  p     = iter->second;

    size_p_vec[size] = p;
  }
  return(size_p_vec);
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
  FILE *output_file;
  
  VShort      nproc             = 4;
  VString     t_input_filename  = NULL;
  VString     hist_file         = NULL;

  static VOptionDescRec program_options[] = {
    {"tin",   VStringRepn,  1, &t_input_filename, VRequiredOpt, NULL, "Threshold input file" },
    {"hist",  VStringRepn,  1, &hist_file,        VRequiredOpt, NULL, "Histogram input file" },
    {"j",     VShortRepn,   1, &nproc,            VOptionalOpt, NULL, "number of processors to use, '0' to use all" }
  };

  VParseFilterCmd( VNumber (program_options), program_options, argc, argv, &input_file, &output_file);

  vector<double> hist = read_hist_file(hist_file);
  double size_p[hist.size()];
  int max_hist_cluster_size = hist.size() - 1;
  cerr << "Maximum cluster in histogramm: " << max_hist_cluster_size << endl;

  for (int i = 0; i < hist.size(); i++) {
    size_p[i] = hist[i];
  }

#ifdef _OPENMP
  // Take care of multiprocessing
  configure_omp(nproc); 
#endif /*OPENMP */
 
  VImage source_image;

  VAttrList attribute_list  = VReadFile(input_file, NULL);
  fclose(input_file);
  
  if(!attribute_list)
    VError("Error reading image");
  
  VAttrListPosn position;

  for (VFirstAttr(attribute_list, &position); VAttrExists(&position); VNextAttr(&position)) {
    if (VGetAttrRepn(&position) != VImageRepn)
      continue;

    // Extract this image
    VGetAttrValue(&position,NULL,VImageRepn,&source_image);
    break;
  }
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

  int number_of_bands   = VImageNBands(source_image);
  int number_of_rows    = VImageNRows(source_image);
  int number_of_columns = VImageNColumns(source_image);

  cerr << "Binarizing image...";
  VImage new_image = VCreateImage(number_of_bands,number_of_rows,number_of_columns, VBitRepn);
  VFillImage(new_image, VAllBands, 0);
  VCopyImageAttrs (source_image, new_image);

  for(int band(0); band < number_of_bands; band++) {
    for(int row(0); row < number_of_rows; row++) {
      for(int column(0); column < number_of_columns; column++) {
        int presence = 0;

        if (fabs(VGetPixel(source_image, band, row, column)) > VGetPixel(t_images[0],band, row, column))
          presence = 1;
        if ((nr_images == 2) && VGetPixel(source_image, band, row, column) < VGetPixel(t_images[1],band, row, column))
          presence = 1;

        VPixel(new_image, band, row, column, VBit) = presence;
      }
    }
  }
  cerr << "done." << endl;

    // Let's count clusters in this picture
  cerr << "Counting clusters...";
  VImage label_image = VCreateImage(number_of_bands, number_of_rows , number_of_columns, VShortRepn);
  int number_of_labels;
  VFillImage(label_image,VAllBands,0);
  label_image = VLabelImage3d(new_image, label_image, 26, VShortRepn, &number_of_labels);

  int cluster_size[number_of_labels+1];

  for (int label = 0; label < number_of_labels; label++) 
    cluster_size[label] = 0;

  for(int band(0); band < number_of_bands; band++) {
    for(int row(0); row < number_of_rows; row++) {
      for(int column(0); column < number_of_columns; column++) {
        short label = VPixel(label_image, band, row, column, VShort);

        if ((label > number_of_labels) || (label < 0)) {
          cerr << "Got unexpected label: " << label << endl;
          exit(-1);
        }
        if (label > 1) cluster_size[label]++;
      }
    }
  }
  cerr << "done." << endl;

  cerr << "Assessing clusters...";
  double cluster_p[number_of_labels+1];
  for (int label = 1; label <= number_of_labels; label++) {
    int cs = cluster_size[label];

    if (cs > max_hist_cluster_size)
      cluster_p[label] = size_p[max_hist_cluster_size-1] / 2;
    else if (cs <= 2)
      cluster_p[label] = 1.0;
    else
      cluster_p[label] = size_p[cs];
  }
  cerr << "done." << endl;

  cerr << "Correcting for multiple comparisons...";
  vector<double> cluster_p_sorted;
  for (int label = 1; label <= number_of_labels; label++) {
    if (cluster_size[label] > 1)
      cluster_p_sorted.push_back(cluster_p[label]);
  }
  sort(cluster_p_sorted.begin(), cluster_p_sorted.end()); 
  /*
  cerr << "P values" << endl;
  BOOST_FOREACH( double p, cluster_p_sorted ) {
    cerr << p << "\t";
  }
  cerr << endl;
  */

  int m = cluster_p_sorted.size();
  cerr << "#Clusters >=2 = " << m << endl;
  vector<double> cluster_delta(m);
  double q = 0.2;

  //cerr << "Deltas" << endl;
  for (int j = 0; j < m; j++) {
    int i = j + 1;
    cluster_delta[j] = 1.0 - pow(1.0 - std::min(1.0, m * q / (m - i + 1.0)),1.0/(m-i+1.0));
    //cerr << cluster_delta[j] << "\t";
  }
  //cerr << endl;

  int j = 0;
  for (; (j < number_of_labels) && (cluster_p_sorted[j] < cluster_delta[j]); j++);
  double p_cut;
  if (j > 0)
    p_cut = cluster_p_sorted[j-1];
  else
    p_cut = 0.0;
  cerr << "done." << endl;
  cerr << "p_cut=" << p_cut << endl;
  /*
*/
  //double p_cut = 1.0;

  cerr << "Writing output image...";
  VImage w_out_image = VCreateImage(number_of_bands,number_of_rows,number_of_columns, VFloatRepn);
  VImage p_out_image = VCreateImage(number_of_bands,number_of_rows,number_of_columns, VFloatRepn);

  VFillImage(w_out_image, VAllBands, 0);
  VFillImage(p_out_image, VAllBands, 0);

  VCopyImageAttrs(source_image, w_out_image);
  VCopyImageAttrs(source_image, p_out_image);

  for(int band(0); band < number_of_bands; band++) {
    for(int row(0); row < number_of_rows; row++) {
      for(int column(0); column < number_of_columns; column++) {
        int label = VGetPixel(label_image, band, row, column);
        if ((label > 0) && (cluster_p[label] <= p_cut)) {
          VPixel(p_out_image, band, row, column, VFloat) = -log10(cluster_p[label]);
          VPixel(w_out_image, band, row, column, VFloat) = VGetPixel(source_image, band, row, column);
        }
      }
    }
  }
  VSetAttr(VImageAttrList(w_out_image),"name",NULL,VStringRepn,"Cluster weight");
  VSetAttr(VImageAttrList(p_out_image),"name",NULL,VStringRepn,"Cluster p");

  VAttrList out_list = VCreateAttrList();
  VAppendAttr(out_list,"image",NULL,VImageRepn, w_out_image);
  VAppendAttr(out_list,"image",NULL,VImageRepn, p_out_image);
  VHistory(VNumber(program_options),program_options,argv[0],&attribute_list,&out_list);
  VWriteFile(output_file, out_list);
  cerr << "Done." << endl;
}

