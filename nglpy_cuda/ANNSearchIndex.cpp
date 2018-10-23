/*
 * ANNSearchIndex.cpp
 *
 *  Created on: Oct 23, 2018
 *      Author: maljovec
 */

#include <ANN/ANN.h>

void ANNSearchIndex::fit(float *X, int N, int D) {
    data = annAllocPts(N, D);
    for(int i = 0; i<N;i++) {
        for(int k=0;k<D;k++) {
            data[i][k] = X[i][k];
        }
    }
    kdTree = new ANNkd_tree(data, N, D);
    nnIdx = new ANNidx[K];
    dists = new ANNdist[K];
}

void ANNSearchIndex::search(int *indices, int N, int *k_indices, float *distances) {
    for(int i = 0; i < N; i++) {
        int id = indices[i];
        kdTree->annkSearch(data[id], K, nnIdx, dists, epsilon);
        for (int k = 0; k < K; k++) {
            int idx = i*K+k;
            k_indices[idx] = nnIdx[k];
            if(distances != NULL) {
                distances[idx] = dists[k];
            }
        }
    }
}

void ANNSearchIndex::search(int startIndex, int count, int *k_indices, float *distances) {
    for(int i = startIndex; i < startIndex+count; i++) {
        kdTree->annkSearch(data[i], K, nnIdx, dists, epsilon);
        for (int k = 0; k < K; k++) {
            int idx = (i-startIndex)*K+k;
            k_indices[idx] = nnIdx[k];
            if(distances != NULL) {
                distances[idx] = dists[k];
            }
        }
    }
}

virtual ANNSearchIndex::~ANNSearchIndex() {
    annDeallocPts(dataPts);
    delete kdTree;
    delete nnIdx;
    delete dists;
}
