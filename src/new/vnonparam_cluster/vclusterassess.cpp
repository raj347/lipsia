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

struct size_p_pair {
  double cluster_size;
  double cluster_p;

  size_p_pair(double &cluster_size_, double &cluster_p_) : cluster_size(cluster_size_), cluster_p(cluster_p_) {

  }

};

struct hist_file_return {
  std::vector<size_p_pair> pos;
  std::vector<size_p_pair> neg;

  hist_file_return(std::vector<size_p_pair> &pos_, std::vector<size_p_pair> &neg_) : pos(pos_), neg(neg_) {

  }

};

struct hist_file_return read_hist_file(VString hist_file) {
  std::vector<size_p_pair> pos;
  std::vector<size_p_pair> neg;

  std::ifstream file(hist_file);
  std::string   line;

  while(std::getline(file, line)) {
    std::stringstream   linestream(line);
    double              val1;
    double              val2;
    double              val3;

    linestream >> val1 >> val2 >> val3;

    if (val3 > 0) {
      pos.push_back(size_p_pair(val1,val2));
    } else {
      neg.push_back(size_p_pair(val1,val2));
    }
  }
  file.close();

  if (pos.size() <= 10) {
    cerr << "Histogramm does not look good. Have only " << pos.size() << " entries" << endl;
    exit(-1);
  }

  return(hist_file_return(pos,neg));
}

VImage labelImage(VImage new_image, int &number_of_labels, int number_of_bands, int number_of_rows, int number_of_columns) {
  VImage label_image = VCreateImage(number_of_bands, number_of_rows , number_of_columns, VShortRepn);
  VFillImage(label_image,VAllBands,0);
  return(VLabelImage3d(new_image, label_image, 6, VShortRepn, &number_of_labels));
}

std::vector<double> assess_image(VImage label_image, int number_of_labels, int number_of_bands, int number_of_rows, int number_of_columns, double max_size, double max_size_p, std::vector<size_p_pair> hist_pair, double q_val) {
  // Let's count clusters in this picture
  cerr << "Counting clusters...";

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
          cluster_size[label-1] += 1;
        }
      }
    }
  }

  std::vector<double> cluster_p(number_of_labels);

  for (int i = 0; i <  number_of_labels; i++) {
    double cs = cluster_size[i];

    if (cs > max_size)
      cluster_p[i] = max_size_p / 2;
    else {
      // Find the largest size, that is smaller than this one
      double local_max_size = 0.0;
      double local_p        = 1.0;
      BOOST_FOREACH( size_p_pair sp, hist_pair) {
        if (sp.cluster_size < cs && sp.cluster_size > local_max_size) {
          local_max_size  = sp.cluster_size;
          local_p         = sp.cluster_p;
        }
      }
      cluster_p[i] = local_p;
    }
  }

  cerr << "Correcting for multiple comparisons...";
  vector<float> cluster_p_sorted;
  //cerr << "Dimensions" << endl;
  for (int label = 0; label < number_of_labels; label++) {
    if (cluster_dimension[label] > 1) {
      cluster_p_sorted.push_back(cluster_p[label]);
    }
  }
  cerr << endl;

  sort(cluster_p_sorted.begin(), cluster_p_sorted.end()); 

  int m = cluster_p_sorted.size();
  cerr << "#Clusters >=2 = " << m << endl;
  vector<float> cluster_delta(m);

  //cerr << "Deltas" << endl;
  //cerr << "q_val used: " << q_val << endl;
  for (int j = 0; j < m; j++) {
    int i = j + 1;
    cluster_delta[j] = 1.0 - pow(1.0 - std::min(1.0, m * q_val / (m - i + 1.0)),1.0/(m-i+1.0));
    //cerr << cluster_delta[j] << "\t";
  }
  //cerr << endl;

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

  for (int label = 0; label < number_of_labels; label++) {
    if (cluster_p[label] > p_cut) {
      cluster_p[label] = 0.0;
    }
  }

  return(cluster_p);
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

  struct hist_file_return pos_neg = read_hist_file(hist_file);

  // Find largest size and its p value
  // Positive clusters
  double max_pos_size   = 0.0;
  double max_pos_p      = 0.0;
  BOOST_FOREACH(size_p_pair sp, pos_neg.pos ) {
    if (sp.cluster_size > max_pos_size) {
      max_pos_size  = sp.cluster_size;
      max_pos_p     = sp.cluster_p;
    }
  }

  // Negative clusters
  double max_neg_size   = 0.0;
  double max_neg_p      = 0.0;
  BOOST_FOREACH(size_p_pair sp, pos_neg.neg ) {
    if (sp.cluster_size > max_neg_size) {
      max_neg_size  = sp.cluster_size;
      max_neg_p     = sp.cluster_p;
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
  VImage pos_image = VCreateImage(number_of_bands,number_of_rows,number_of_columns, VBitRepn);
  VImage neg_image = VCreateImage(number_of_bands,number_of_rows,number_of_columns, VBitRepn);

  VFillImage(pos_image, VAllBands, 0);
  VFillImage(neg_image, VAllBands, 0);

  VCopyImageAttrs (source_image, pos_image);
  VCopyImageAttrs (source_image, neg_image);

  for(int band(0); band < number_of_bands; band++) {
    for(int row(0); row < number_of_rows; row++) {
      for(int column(0); column < number_of_columns; column++) {
        // Positive cluster
        if (VGetPixel(source_image, band, row, column) > VGetPixel(t_images[0],band, row, column)) {
          VPixel(pos_image, band, row, column, VBit) = 1;
        }
        // Negative cluster (only in two sided tests)
        if ((nr_images == 2) && VGetPixel(source_image, band, row, column) < VGetPixel(t_images[1],band, row, column)) {
          VPixel(neg_image, band, row, column, VBit) = 1;
        }
      }
    }
  }
  cerr << "done." << endl;

  int number_of_pos_labels;
  int number_of_neg_labels;

  VImage pos_labels = labelImage(pos_image, number_of_pos_labels, number_of_bands, number_of_rows, number_of_columns);
  VImage neg_labels = labelImage(neg_image, number_of_neg_labels, number_of_bands, number_of_rows, number_of_columns);

  std::vector<double> pos_cluster_p = assess_image(pos_labels, number_of_pos_labels, number_of_bands, number_of_rows, number_of_columns, max_pos_size, max_pos_p, pos_neg.pos, q_val);
  std::vector<double> neg_cluster_p = assess_image(neg_labels, number_of_neg_labels, number_of_bands, number_of_rows, number_of_columns, max_neg_size, max_neg_p, pos_neg.neg, q_val);

  /*
  for(int i = 0; i < pos_cluster_p.size(); i++) {
    if (pos_cluster_p[i] > 0)
      std::cerr << "Cluster " << i << " has a good p value of " << pos_cluster_p[i] << std::endl;
  }
  for(int i = 0; i < neg_cluster_p.size(); i++) {
    if (neg_cluster_p[i] > 0)
      std::cerr << "Cluster " << i << " has a good p value of " << neg_cluster_p[i] << std::endl;
  }
  */

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
        // First positive
        int pos_label = VGetPixel(pos_labels, band, row, column);

        if ((pos_label > 0) && (pos_cluster_p[pos_label-1] > 0.0)) {
          VPixel(p_out_image, band, row, column, VFloat) = -log10(pos_cluster_p[pos_label-1]);
          VPixel(w_out_image, band, row, column, VFloat) = VGetPixel(source_image, band, row, column);
          VPixel(l_out_image, band, row, column, VShort) = pos_label;
        }

        int neg_label = VGetPixel(neg_labels, band, row, column);

        if ((neg_label > 0) && (neg_cluster_p[neg_label-1] > 0.0)) {
          VPixel(p_out_image, band, row, column, VFloat) = -log10(neg_cluster_p[neg_label-1]);
          VPixel(w_out_image, band, row, column, VFloat) = VGetPixel(source_image, band, row, column);
          VPixel(l_out_image, band, row, column, VShort) = -neg_label;
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

