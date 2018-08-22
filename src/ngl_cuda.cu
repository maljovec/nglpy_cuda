#include "ngl_cuda.cuh"
#include <cstdio>
#include <vector>
#include <map>
#include <algorithm>

#define cudaErrchk(ans) { GPUAssert((ans), __FILE__, __LINE__); }
inline void GPUAssert(cudaError_t code, const char *file, int line,
                      bool abort=true) {
	if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
                line);
		if (abort) exit(code);
	}
}

namespace nglcu {
    dim3 block_size(32, 32);
    dim3 grid_size(4, 4);

    __global__
    void prune_discrete_d(float *X, int *edgesIn, const int N, const int D,
                        const int K, const int steps, float *erTemplate,
                        int *edgesOut) {
        int index_x = blockIdx.x * blockDim.x + threadIdx.x;
        int stride_x = blockDim.x * gridDim.x;

        int index_y = blockIdx.y * blockDim.y + threadIdx.y;
        int stride_y = blockDim.y * gridDim.y;

        // References to points in X
        float *p, *q, *r;

        //TODO: Fix this to be dynamically allocated
        // Computed vectors representing the edge under test pq and the vector from
        // one end point to a third point r (We will iterate over all possible r's)
        float pq[10] = {};
        float pr[10] = {};

        // Different iterator/indexing variables i, j, and n are rows in X
        // representing p, q, and r, respectively
        // k is the nearest neighbor, d is the dimension
        int i, j, k, k2, d, n;

        // t is the parameterization of the projection of pr onto pq
        // In layman's terms, this is the length of the shadow pr casts onto pq
        // lookup is the
        float t;
        int lookup;

        // Some other temporary variables
        float length_squared;
        float squared_distance_to_edge;
        float minimum_allowable_distance;

        for (k = index_y; k < K; k += stride_y) {
            for (i = index_x; i < N; i += stride_x) {

                p = &(X[D*i]);
                j = edgesIn[K*i+k];
                q = &(X[D*j]);

                length_squared = 0;
                for(d = 0; d < D; d++) {
                    pq[d] = p[d] - q[d];
                    length_squared += pq[d]*pq[d];
                }
                // A point should not be connected to itself
                if(length_squared == 0) {
                    edgesOut[K*i+k] = -1;
                    continue;
                }

                // for(n = 0; n < N; n++) {
                for(k2 = 0; k2 < 2*K; k2++) {
                    n = (k2 < K) ? edgesIn[K*i+k2] : edgesIn[K*j+(k2-K)];
                    r = &(X[D*n]);

                    t = 0;
                    for(d = 0; d < D; d++) {
                        pr[d] = p[d] - r[d];
                        t += pr[d]*pq[d];
                    }

                    t /= length_squared;
                    lookup = __float2int_rd(abs(steps * (2 * t - 1))+0.5);
                    if (lookup >= 0 && lookup <= steps) {
                        squared_distance_to_edge = 0;
                        for(d = 0; d < D; d++) {
                            squared_distance_to_edge += (pr[d] - pq[d]*t)*(pr[d] - pq[d]*t);
                        }
                        minimum_allowable_distance = sqrt(length_squared)*erTemplate[lookup];

                        if(sqrt(squared_distance_to_edge) < minimum_allowable_distance) {
                            edgesOut[K*i+k] = -1;
                            break;
                        }
                    }
                }
            }
        }
    }

    __global__
    void prune_d(float *X, int *edgesIn, const int N, const int D, const int K,
                float lp, float beta, int *edgesOut) {
        int index_x = blockIdx.x * blockDim.x + threadIdx.x;
        int stride_x = blockDim.x * gridDim.x;

        int index_y = blockIdx.y * blockDim.y + threadIdx.y;
        int stride_y = blockDim.y * gridDim.y;

        float *p, *q, *r;

        float pq[10] = {};
        float pr[10] = {};

        int i, j, k, k2, d, n;
        float t;

        float length_squared;
        float squared_distance_to_edge;
        float minimum_allowable_distance;

        ////////////////////////////////////////////////////////////
        float xC, yC, radius, y;
        ////////////////////////////////////////////////////////////

        for (k = index_y; k < K; k += stride_y) {
            for (i = index_x; i < N; i += stride_x) {
                p = &(X[D*i]);
                j = edgesIn[K*i+k];
                q = &(X[D*j]);

                length_squared = 0;
                for(d = 0; d < D; d++) {
                    pq[d] = p[d] - q[d];
                    length_squared += pq[d]*pq[d];
                }
                // A point should not be connected to itself
                if(length_squared == 0) {
                    edgesOut[K*i+k] = -1;
                    continue;
                }

                // for(n = 0; n < N; n++) {
                for(k2 = 0; k2 < 2*K; k2++) {
                    n = (k2 < K) ? edgesIn[K*i+k2] : edgesIn[K*j+(k2-K)];
                    r = &(X[D*n]);

                    // t is the parameterization of the projection of pr onto pq
                    // In layman's terms, this is the length of the shadow pr casts onto pq
                    t = 0;
                    for(d = 0; d < D; d++) {
                        pr[d] = p[d] - r[d];
                        t += pr[d]*pq[d];
                    }

                    t /= length_squared;

                    if (t > 0 && t < 1) {
                        squared_distance_to_edge = 0;
                        for(d = 0; d < D; d++) {
                            squared_distance_to_edge += (pr[d] - pq[d]*t)*(pr[d] - pq[d]*t);
                        }

                        ////////////////////////////////////////////////////////////
                        // ported from python function, can possibly be improved
                        // in terms of performance
                        xC = 0;
                        yC = 0;

                        if (beta <= 1) {
                            radius = 1. / beta;
                            yC = powf(powf(radius, lp) - 1, 1. / lp);
                        }
                        else {
                            radius = beta;
                            xC = 1. - beta;
                        }
                        t = fabs(2*t-1);
                        y = powf(powf(radius, lp) - powf(t-xC, lp), 1. / lp) - yC;
                        minimum_allowable_distance = 0.5*y*sqrt(length_squared);

                        //////////////////////////////////////////////////////////
                        if(sqrt(squared_distance_to_edge) < minimum_allowable_distance) {
                            edgesOut[K*i+k] = -1;
                            break;
                        }
                    }
                }
            }
        }
    }

    float min_distance_from_edge(float t, float beta, float p) {
        float xC = 0;
        float yC = 0;
        float r;

        if (t > 1) {
            return 0;
        }
        if (beta <= 1) {
            r = 1. / beta;
            yC = powf(powf(r, p) - 1, 1. / p);
        }
        else {
            r = beta;
            xC = 1 - beta;
        }
        float y = powf(powf(r, p) - powf(t-xC, p), 1. / p) - yC;
        return 0.5*y;
    }

    void create_template(float * data, float beta, int p, int steps) {
        if (p < 0) {
            if (beta >= 1) {
                for (int i = 0; i <= steps; i++) {
                    data[i] = beta / 2.;
                }
            }
            else {
                for (int i = 0; i <= steps; i++) {
                    data[i] = 0.;
                }
            }
        }
        else {
            for (int i = 0; i <= steps; i++) {
                data[i] = min_distance_from_edge(float(i)/steps, beta, p);
            }
        }
    }

    void prune_discrete(const int N, const int D, const int K, const int steps,
                    const float beta, const float p, float *X, int *edges) {
        float erTemplate[steps];
        create_template(erTemplate, beta, p, steps);
        return prune_discrete(N, D, K, steps, erTemplate, X, edges);
    }

    void prune_discrete(const int N, const int D, const int K, const int steps,
                    float *erTemplate, float *X, int *edges) {
        float *x_d;
        int *edgesIn_d;
        int *edgesOut_d;
        float *erTemplate_d;
        cudaErrchk(cudaMallocManaged(&x_d, N*D*sizeof(float)));
        cudaErrchk(cudaMallocManaged(&edgesIn_d, N*K*sizeof(int)));
        cudaErrchk(cudaMallocManaged(&edgesOut_d, N*K*sizeof(int)));
        cudaErrchk(cudaMallocManaged(&erTemplate_d, (steps)*sizeof(float)));

        memcpy(x_d, X, N*D*sizeof(float));
        memcpy(edgesIn_d, edges, N*K*sizeof(float));
        memcpy(edgesOut_d, edges, N*K*sizeof(float));
        memcpy(erTemplate_d, erTemplate, (steps)*sizeof(float));

        prune_discrete_d<<<grid_size, block_size>>>(x_d, edgesIn_d, N, D, K, steps,
                                                    erTemplate_d, edgesOut_d);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            printf("Error: %s\n", cudaGetErrorString(err));
        cudaDeviceSynchronize();

        memcpy(edges, edgesOut_d, N*K*sizeof(float));

        cudaFree(x_d);
        cudaFree(edgesIn_d);
        cudaFree(edgesOut_d);
        cudaFree(erTemplate_d);
    }

    void prune(const int N, const int D, const int K, float lp, float beta,
        float *X, int *edges) {
        float *x_d;
        int *edgesIn_d;
        int *edgesOut_d;
        cudaMallocManaged(&x_d, N*D*sizeof(float));
        cudaMallocManaged(&edgesIn_d, N*K*sizeof(int));
        cudaMallocManaged(&edgesOut_d, N*K*sizeof(int));

        memcpy(x_d, X, N*D*sizeof(float));
        memcpy(edgesIn_d, edges, N*K*sizeof(float));
        memcpy(edgesOut_d, edges, N*K*sizeof(float));

        prune_d<<<grid_size, block_size>>>(x_d, edgesIn_d, N, D, K, lp, beta,
                                        edgesOut_d);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            printf("Error: %s\n", cudaGetErrorString(err));
        cudaDeviceSynchronize();

        memcpy(edges, edgesOut_d, N*K*sizeof(float));

        cudaFree(x_d);
        cudaFree(edgesIn_d);
        cudaFree(edgesOut_d);
    }

    vector_edge get_edge_list(int *edges, const int N, const int K) {
        int i, k;
        vector_edge edge_list;
        for(i = 0; i < N; i++) {
            for(k = 0; k < K; k++) {
                if (edges[i*K+k] != -1) {
                    edge_list.push_back(std::make_pair(i, edges[i*K+k]));
                }
            }
        }
        return edge_list;
    }

    void print_cuda_info() {
        struct cudaDeviceProp properties;
        cudaGetDeviceProperties(&properties, 0);
        fprintf(stderr, "using %d multiprocessors\n", properties.multiProcessorCount);
        fprintf(stderr, "max threads per processor: %d\n", properties.maxThreadsPerMultiProcessor);
        fprintf(stderr, "Grid Size: %dx%d\n", grid_size.x, grid_size.y);
        fprintf(stderr, "Block Size: %dx%d\n", block_size.x, block_size.y);
    }
}
