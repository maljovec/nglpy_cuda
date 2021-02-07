#ifndef NGLCUDA_CUH
#define NGLCUDA_CUH
#include <cstdlib>

namespace nglcu{
    void create_template(float * data, float beta=1, int p=2, int steps=100);
    float min_distance_from_edge(float t, float beta, float p);
    void associate_probability(float *X, int *edges, float *probabilities, int *indices,
                               int N, int D, int M, int K, float steepness=3,
                               bool relaxed=false, float beta=1, float lp=2,
                               int count=-1);
    void prune_discrete(float *X, int *edges, int *indices, int N, int D, int M,
                        int K, float *erTemplate=NULL,
                        int steps=100, bool relaxed=false, float beta=1,
                        float lp=2, int count=-1);
    void prune(float *X, int *edges, int *indices, int N, int D, int M, int K,
               bool relaxed=false, float beta=1, float lp=2, int count=-1);
    void prune_yao(float *X, float *bisectors, int *edges, int *indices, int N,
                   int D, int M, int K, int numSectors, int numPointsPerSector,
                   int count);
    void print_cuda_info();
    size_t get_available_device_memory();
}

#endif //NGLCUDA_CUH
