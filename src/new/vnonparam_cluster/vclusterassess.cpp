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
#include <utility>

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
using std::make_pair;
using std::pair;

#define EPSILON 1e-8

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

std::vector< std::pair< double , double > > read_hist_file(VString hist_file) {
  std::vector< std::pair< double , double > > mapping_size_p;

  std::ifstream file(hist_file);
  std::string   line;

  int max_cluster_size = 0;
  while(std::getline(file, line)) {
    std::stringstream   linestream(line);
    double              val1;
    double              val2;

    linestream >> val1 >> val2;
    mapping_size_p.push_back(make_pair (val1,val2));

    if (val1 > max_cluster_size)
      max_cluster_size = val1;
  }
  file.close();

  // Sanity check
  if (max_cluster_size > 1e6) {
    cerr << "Cluster size seems to be too large" << endl;
    exit(-1);
  }

  if (mapping_size_p.size() <= 10) {
    cerr << "Histogramm does not look good. Have only " << mapping_size_p.size() << " entries" << endl;
    exit(-1);
  }

  return(mapping_size_p);
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
  VFloat      q_val             = 0.2;

  static VOptionDescRec program_options[] = {
    {"tin",   VStringRepn,  1, &t_input_filename, VRequiredOpt, NULL, "Threshold input file" },
    {"hist",  VStringRepn,  1, &hist_file,        VRequiredOpt, NULL, "Histogram input file" },
    {"q",     VFloatRepn,   1, &q_val,            VOptionalOpt, NULL, "q value in classical fdr" },
    {"j",     VShortRepn,   1, &nproc,            VOptionalOpt, NULL, "number of processors to use, '0' to use all" }
  };

  VParseFilterCmd( VNumber (program_options), program_options, argc, argv, &input_file, &output_file);

#ifdef _OPENMP
  // Take care of multiprocessing
  configure_omp(nproc); 
#endif /*OPENMP */

  vector< pair<double,double> > hist = read_hist_file(hist_file);
  typedef pair<double,double> pair_double_double_type;
  // Find largest size and its p value
  double max_size   = 0.0;
  double max_size_p = 0.0;
  BOOST_FOREACH( pair_double_double_type mapping, hist) {
    if (mapping.first > max_size) {
      max_size    = mapping.first;
      max_size_p  = mapping.second;
    }

  }

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

  if (nr_images == 2) {
    cerr << "Two sided test" << endl;
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

        if (VGetPixel(source_image, band, row, column) > VGetPixel(t_images[0],band, row, column))
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
  //label_image = VLabelImage3d(new_image, label_image, 26, VShortRepn, &number_of_labels);
  label_image = VLabelImage3d(new_image, label_image, 6, VShortRepn, &number_of_labels);

  vector<double>  cluster_size(number_of_labels);
  vector<int>     cluster_dimension(number_of_labels);

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
          if (VGetPixel(source_image, band, row, column) > VGetPixel(t_images[0], band, row, column)) {
            //cluster_size[label-1] += (VGetPixel(source_image, band, row, column) / VGetPixel(t_images[0],band, row, column) -1);
            cluster_size[label-1] += 1;
          } else if ((nr_images == 2) && VGetPixel(source_image, band, row, column) < VGetPixel(t_images[1],band, row, column)) {
            //cluster_size[label-1] += (VGetPixel(source_image, band, row, column) / VGetPixel(t_images[1],band, row, column) -1);
            cluster_size[label-1] += 1;
          }
        }
      }
    }
  }
  cerr << "done." << endl;

  cerr << "Assessing clusters...";
  cerr << "number_of_labels=" << number_of_labels << endl;
  double cluster_p[number_of_labels];

  for (int i = 0; i <  number_of_labels; i++) {
    double cs = cluster_size[i];

    cerr << "i=" << i << "cs=" << cs << endl;

    if (cs > max_size)
      cluster_p[i] = max_size_p / 2;
    else {
      // Find the largest size, that is smaller than this one
      double local_max_size = 0.0;
      double local_p        = 1.0;
      BOOST_FOREACH( pair_double_double_type mapping, hist) {
        if (mapping.first < cs && mapping.first > local_max_size) {
          local_max_size  = mapping.first;
          local_p         = mapping.second;
        }
      }
      cluster_p[i] = local_p;
    }
    cerr << "p=" << cluster_p[i] << endl;
    cerr << "Label " << (i+1) << " has size=" <<  cs << " and gets p=" << cluster_p[i] << endl;
  }
  cerr << "done." << endl;

  cerr << "Correcting for multiple comparisons...";
  vector<float> cluster_p_sorted;
  cerr << "Dimensions" << endl;
  for (int label = 0; label < number_of_labels; label++) {
    if (cluster_dimension[label] > 1) {
      cerr << cluster_dimension[label] << "(" << cluster_p[label] << ")" << "\t";
      cluster_p_sorted.push_back(cluster_p[label]);
    }
  }
  cerr << endl;

  sort(cluster_p_sorted.begin(), cluster_p_sorted.end()); 
  cerr << "P values" << endl;
  BOOST_FOREACH( float p, cluster_p_sorted ) {
    cerr << p << "\t";
  }
  cerr << endl;

  int m = cluster_p_sorted.size();
  cerr << "#Clusters >=2 = " << m << endl;
  vector<float> cluster_delta(m);

  cerr << "Deltas" << endl;
  cerr << "q_val used: " << q_val << endl;
  for (int j = 0; j < m; j++) {
    int i = j + 1;
    cluster_delta[j] = 1.0 - pow(1.0 - std::min(1.0, m * q_val / (m - i + 1.0)),1.0/(m-i+1.0));
    cerr << cluster_delta[j] << "\t";
  }
  cerr << endl;

  int j = 0;
  for (; (j < m) && (cluster_p_sorted[j] < cluster_delta[j]); j++);
  float p_cut;
  if (j > 0)
    if (j < m) 
      p_cut = cluster_p_sorted[j-1];
    else
      p_cut = 1.0;
  else
    p_cut = 0.0;
  cerr << "done." << endl;
  cerr << "p_cut=" << p_cut << endl;
  /*
*/
  //double p_cut = 1.0;
  for (int i = 0; i < number_of_labels; i++) {
    if ((cluster_p[i] - p_cut) <= EPSILON) {
    //if (1) {
      cerr << "Label " << (i+1) << " is in (dimension=" << cluster_dimension[i] << ")" << endl;
    }
  }

  cerr << "Writing output image...";
  VImage w_out_image = VCreateImage(number_of_bands,number_of_rows,number_of_columns, VFloatRepn);
  VImage p_out_image = VCreateImage(number_of_bands,number_of_rows,number_of_columns, VFloatRepn);
  VImage l_out_image = VCreateImage(number_of_bands,number_of_rows,number_of_columns, VShortRepn);

  VFillImage(w_out_image, VAllBands, 0);
  VFillImage(p_out_image, VAllBands, 0);
  VFillImage(l_out_image, VAllBands, 0);

  VCopyImageAttrs(source_image, w_out_image);
  VCopyImageAttrs(source_image, p_out_image);
  VCopyImageAttrs(source_image, l_out_image);

  for(int band(0); band < number_of_bands; band++) {
    for(int row(0); row < number_of_rows; row++) {
      for(int column(0); column < number_of_columns; column++) {
        int label = VGetPixel(label_image, band, row, column);
        if ((label > 0) && (cluster_p[label-1] - p_cut) <= EPSILON) {
        //if (label > 0) {
          VPixel(p_out_image, band, row, column, VFloat) = -log10(cluster_p[label-1]);
          VPixel(w_out_image, band, row, column, VFloat) = VGetPixel(source_image, band, row, column);
          VPixel(l_out_image, band, row, column, VShort) = label;
        }
      }
    }
  }
  VSetAttr(VImageAttrList(w_out_image),"name",NULL,VStringRepn,"Cluster weight");
  VSetAttr(VImageAttrList(p_out_image),"name",NULL,VStringRepn,"Cluster p");
  VSetAttr(VImageAttrList(l_out_image),"name",NULL,VStringRepn,"Cluster labels");

  VAttrList out_list = VCreateAttrList();
  VAppendAttr(out_list,"image",NULL,VImageRepn, w_out_image);
  VAppendAttr(out_list,"image",NULL,VImageRepn, p_out_image);
  VAppendAttr(out_list,"image",NULL,VImageRepn, l_out_image);
  VHistory(VNumber(program_options),program_options,argv[0],&attribute_list,&out_list);
  VWriteFile(output_file, out_list);
  cerr << "Done." << endl;
}

