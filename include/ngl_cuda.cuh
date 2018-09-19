#ifndef NGLCUDA_CUH
#define NGLCUDA_CUH

#include <vector>

namespace nglcu{
    typedef typename std::vector<std::pair<int, int> > vector_edge;

    vector_edge get_edge_list(int *edges, const int N, const int K);
    void create_template(float * data, float beta=1, int p=2, int steps=100);
    float min_distance_from_edge(float t, float beta, float p);
    void prune_discrete(float *X, int *edges, const int N, const int D,
                        const int K, float *erTemplate=NULL,
                        const int steps=100, bool relaxed=false, float beta=1,
                        float lp=2);
    void prune(float *X, int *edges, const int N, const int D, const int K,
               bool relaxed=false, float beta=1, float lp=2);

    void print_cuda_info();
}

#endif //NGLCUDA_CUH
