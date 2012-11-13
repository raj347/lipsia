/**
 * @file Threshold.h
 *  
 * @author Tilo Buschmann, Johannes Stelzer
 * @date  23.10.2012
 */

#ifndef THRESHOLD_H_
#define THRESHOLD_H_

#include <boost/concept_check.hpp>
#include <boost/multi_array.hpp>
#include <boost/assign.hpp>
#include <boost/progress.hpp>
#include <boost/foreach.hpp>

#include <vector>

using std::vector;

class Threshold {
  public:
    Threshold(  );

    virtual ~Threshold();

    void    printConfiguration();
    boost::multi_array<float, 4> calculate( boost::multi_array<float, 4> &pool, float p, bool is_two_sided );

};

#endif // THRESHOLD_H_
