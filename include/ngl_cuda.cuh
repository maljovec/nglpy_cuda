#ifndef NGLCUDA_CUH
#define NGLCUDA_CUH

namespace nglcu{
    typedef typename std::vector<std::pair<int, int> > vector_edge;

    vector_edge get_edge_list(int *edges, const int N, const int K);
    void create_template(float * data, float beta=1, int p=2, int steps=100);
    float min_distance_from_edge(float t, float beta, float p);
    void prune_discrete(const int N, const int D, const int K, const int steps,
                        float *erTemplate, float *X, int *edges);
    void prune_discrete(const int N, const int D, const int K, const int steps,
                        float beta, float p, float *X, int *edges);
    void prune(const int N, const int D, const int K, float lp, float beta,
               float *X, int *edges);

    void print_cuda_info();
    dim3 block_size(32, 32);
    dim3 grid_size(4, 4);
}

#endif //NGLCUDA_CUH