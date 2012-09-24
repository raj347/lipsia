/**
 * @file PCA.h
 *
 *  Created on: 26.07.2012
 *  
 * @author Tilo Buschmann
 */

#ifndef PCA_H
#define PCA_H

#include <cfloat>
#include <vector>
#include <string>

#include "boost/multi_array.hpp"

// GSL
#include <gsl/gsl_matrix.h>

using std::vector;

typedef boost::multi_array<double, 2> matrix_2d;

class PrComp {

public:
    PrComp(int m, int n, gsl_matrix *,matrix_2d,int);
    gsl_matrix *getRotation();
    int getP();
    matrix_2d getX();
    void invert(boost::multi_array<double,1> &weights, boost::multi_array<double,1> &inverted_weight);
    void invert_permutation(boost::multi_array<double,1> &inverted_permutated_voxel_weight, boost::multi_array<double,2> &weights, int feature_index, int permutations);

private:
    void setX(matrix_2d);

    matrix_2d x_;
    int m_,n_,p_;
    gsl_matrix *rotation_;
};

class PCA {
public:
  PCA();
  virtual ~PCA();
  void    printConfiguration();
  static PrComp prcomp(matrix_2d A);

private:

};

#endif // PCA_H 
