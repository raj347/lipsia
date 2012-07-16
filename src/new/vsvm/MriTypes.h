/**
 * @file MriTypes.h
 * 
 * Some common data types that I need in multiple headers at once
 * 
 * @author Tilo Buschmann
 * 
 */
#ifndef MRITYPES_H
#define MRITYPES_H

/**
 * A two-dimensional array that contains features per sample (i.e. it is samples * features)
 */
typedef boost::multi_array<double, 2>     sample_features_array_type;

/**
 * Two-dimensional array that contains a number of permutations in the form number_of_permutations x samples
 */
typedef boost::multi_array<int, 2> permutations_array_type;

/**
 * Contains all features for every voxel for every sample, but with all dimensions intact:
 * sample * band * row * column * feature
 * 
 * i.e. this data representation allows for more than one feature per voxel
 */
typedef boost::multi_array<double, 5> sample_3d_array_type; // sample,band,row,column,feature

/**
 * Representation of validities in 3D: band * row * column
 * 
 * Generated as a result by searchlighsvm
 */
typedef boost::multi_array<double, 3> sample_validity_array_type;

/**
 * Array of exactly three elements, representing 3D-Coordinates band,row,column 
 */
typedef boost::array<int,3>           coords_3d;


/**
 * Validities per permutation, careful: band * row * column * permutation
 */
typedef boost::multi_array< float, 4> permutated_validities_type;

#endif // MRITYPES_H