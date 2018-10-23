/*
 * ANNSearchIndex.h
 *
 *  Created on: Oct 23, 2018
 *      Author: maljovec
 */

#ifndef NGL_ANN_SEARCH_INDEX_H
#define NGL_ANN_SEARCH_INDEX_H

#include "SearchIndex.h"
#include <ANN/ANN.h>

class ANNSearchIndex : public SearchIndex
{
public:
  ANNSearchIndex(int K, float epsilon) : K(K), epsilon(epsilon) { }
  ~ANNSearchIndex();
  void fit(float *X, int N, int D);
  void search(int *indices, int N, int *k_indices, float *distances=NULL);
  void search(int startIndex, int count, int *k_indices, float *distances=NULL);
private:
    ANNpointArray data;
    ANNkd_tree* kdTree;
    ANNidxArray nnIdx;
    ANNdistArray dists;
    int K;
    float epsilon;
};

#endif
