/**
 * @file PCA.cpp
 *
 *  Created on: 26.07.2012
 *  
 * @author Tilo Buschmann
 * 
 */

#include <iostream>
#include <fstream>

#include <boost/foreach.hpp>

// GSL headers
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>

#include "PCA.h"

using std::ofstream;
using std::cerr;
using std::cout;
using std::endl;
using std::vector;

/**
 * PCA implements SVD-based PCA
 * 
 */
PCA::PCA()
{
}

/**
 * Empty Destructor
 * 
 */
PCA::~PCA() {
}

/**
 * Print out all member variables
 */
void PCA::printConfiguration() {
}

PrComp PCA::prcomp(matrix_2d A) {
  // Get shape of array (m*n)
  int m = A.shape()[0];
  int n = A.shape()[1];

  cout << "m=" << m << " n=" << n << endl;

  // Scale Matrix A along features
  for (int i = 0; i < n; i++) {
    double sum = 0.0;
    for (int j = 0; j < m; j++)
      sum += A[j][i];

    double mean = sum / m;

    for (int j = 0; j < m; j++)
      A[j][i] -= mean;
  }

  // Transform to gsl Matrix
  gsl_matrix *gA = gsl_matrix_alloc(m,n);

  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++)
      gsl_matrix_set(gA, i, j, A[i][j]);


  // Because m < n, we need to wiggle around a tiny bit
  gsl_matrix *gA_t = gsl_matrix_alloc (n,m);
  gsl_matrix_transpose_memcpy(gA_t, gA); 

  gsl_matrix *V     = gsl_matrix_alloc(m,m);
  gsl_vector *S     = gsl_vector_alloc(m);
  gsl_vector *work  = gsl_vector_alloc(m);
  gsl_matrix *X     = gsl_matrix_alloc(m,m);

  gsl_linalg_SV_decomp_mod(gA_t, X, V, S, work);
  //gsl_linalg_SV_decomp(gA_t, V, S, work);
  //gsl_linalg_SV_decomp_jacobi(gA_t, V, S);

  gsl_vector_free(work);

  /*
  // Let's print out vector S
  for (int s = 0; s < m; s++) {
    cout << gsl_vector_get(S,s) << "\t";
  }
  cout << endl;
  // Let's print out standard deviation
  for (int s = 0; s < m; s++) {
    cout << gsl_vector_get(S,s) / sqrt(m - 1) << "\t";
  }
  cout << endl;
  */

  // The rotation matrix is actually U, because:
  // A=U S V^T, A^T = V S U^T

  // Reuse X for A %*% V
  gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,gA,gA_t,0.0,X); 

  // Let's check, if we want to use all principal components
  int p = m;
  for (int s = 0; s < m; s++) {
    if (gsl_vector_get(S,s) < 1e-6) {
      p = s;
      break;
    }
  }

  matrix_2d x(boost::extents[m][m]);
  for (int i = 0; i < m; i++)
    for (int j = 0; j < m; j++)
      x[i][j] = gsl_matrix_get(X, i, j);

  PrComp result(m,n,gA_t,x,p);
  
  gsl_vector_free(S);
  gsl_matrix_free(V);
  gsl_matrix_free(X);
  gsl_matrix_free(gA);

  return result;
}

PrComp::PrComp(int m, int n, gsl_matrix *rotation,matrix_2d x,int p) : m_(m), n_(n), rotation_(rotation),p_(p) {
  setX(x);
}

void PrComp::setX(matrix_2d x) {
  std::vector<size_t> ex;
  const size_t* shape = x.shape();
  ex.assign(shape,shape+x.num_dimensions());
  x_.resize(ex);
  x_ = x;
}

matrix_2d PrComp::getX() {
  return x_;
}

int PrComp::getP() {
  return(p_);
}

gsl_matrix *PrComp::getRotation() {
  return rotation_;
}

vector<double> PrComp::invert(vector<double> weight) {
  int p = weight.size();
  vector<double> inverted_weight(n_,0.0);

  for (int i = 0; i < n_; i++)
    for (int j = 0; j < p; j++)
      inverted_weight[i] += weight[j] * gsl_matrix_get(rotation_,i,j);

  return inverted_weight;
}

