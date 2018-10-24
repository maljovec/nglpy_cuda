/*
 * FAISSSearchIndex.h
 *
 *  Created on: Oct 23, 2018
 *      Author: maljovec
 */

#ifndef NGL_FAISS_SEARCH_INDEX_H
#define NGL_FAISS_SEARCH_INDEX_H

#include "SearchIndex.h"

class FAISSSearchIndex : public SearchIndex
{
public:
  FAISSSearchIndex(int K);
  ~FAISSSearchIndex();
  void fit(float *X, int N, int D);
  void search(int *indices, int N, int *k_indices, float *distances=NULL);
  void search(int startIndex, int count, int *k_indices, float *distances=NULL);
};

#endif
