#ifndef MRITYPES_H
#define MRITYPES_H

typedef boost::multi_array<double, 2>     sample_features_array_type;
typedef sample_features_array_type::index sample_features_array_type_index;

typedef boost::multi_array<double, 2> permutations_array_type;

typedef boost::multi_array<double, 5> sample_3d_array_type; // sample,band,row,column,feature
typedef boost::multi_array<double, 3> sample_validity_array_type;
typedef boost::array<int,3>           coords_3d;

typedef boost::multi_array< float, 4> permutated_validities_type;

#endif // MRITYPES_H