#ifndef NGLCUDA_CUH
#define NGLCUDA_CUH

#include <vector>

namespace nglcu{
    typedef typename std::vector<std::pair<int, int> > vector_edge;

    vector_edge get_edge_list(int *edges, int N, int K);
    void create_template(float * data, float beta=1, int p=2, int steps=100);
    float min_distance_from_edge(float t, float beta, float p);
    void prune_discrete(float *X, int *edges, int N, int D,
                        int K, float *erTemplate=NULL,
                        int steps=100, bool relaxed=false, float beta=1,
                        float lp=2, int count=-1);
    void prune(float *X, int *edges, int N, int D, int K,
               bool relaxed=false, float beta=1, float lp=2, int count=-1);

    void print_cuda_info();
    size_t get_available_device_memory();
}

#endif //NGLCUDA_CUH
