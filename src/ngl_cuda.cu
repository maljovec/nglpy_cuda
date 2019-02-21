#include "ngl_cuda.cuh"
#include "utils.cuh"
#include <cstdio>
#include <map>
#include <chrono>
#include <iostream>

namespace nglcu {

    __device__
    float logistic_function(float x, float r, float steepness=3.) {
        // P(0) = 0.047... ~ 0.0
        // P(r) = 0.5
        // P(3*r) = 0.997... ~ 1.0
        float k = steepness / r;
        return 1. / (1. + expf(-k*(x - r)));
    }

    __global__
    void prune_discrete_d(float *X, int *edgesIn, int N, int D,
                        int K, float *erTemplate, int steps,
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
    void prune_discrete_relaxed_d(float *X, int *edgesIn, int N,
                                  int D, int K, float *erTemplate,
                                  int steps, int *edgesOut) {
        int index_x = blockIdx.x * blockDim.x + threadIdx.x;
        int stride_x = blockDim.x * gridDim.x;

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

        for (k = 0; k < K; k++) {
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

                // This loop presumes that all nearer neighbors have
                // already been processed
                for(k2 = 0; k2 < k; k2++) {
                    n = edgesOut[K*i+k2];
                    if (n == -1){
                        continue;
                    }
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
    void prune_d(float *X, int *edgesIn, int N, int D, int K,
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

    __global__
    void prune_relaxed_d(float *X, int *edgesIn, int N, int D,
                         int K, float lp, float beta, int *edgesOut) {
        int index_x = blockIdx.x * blockDim.x + threadIdx.x;
        int stride_x = blockDim.x * gridDim.x;

        // We should use a 1D structure for this since we need to guarantee
        // that other points have already been processed
        // int index_y = blockIdx.y * blockDim.y + threadIdx.y;
        // int stride_y = blockDim.y * gridDim.y;

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

        for (i = index_x; i < N; i += stride_x) {
            for (k = 0; k < K; k++) {
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

                // This loop presumes that all nearer neighbors have
                // already been processed
                for(k2 = 0; k2 < k; k2++) {
                    n = edgesOut[K*i+k2];
                    if (n == -1){
                        continue;
                    }
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

    __global__
    void probability_d(float *X,
                       int *edgesIn,
                       int N,
                       int D,
                       int K,
                       float lp,
                       float beta,
                       float steepness,
                       float *probabilities) {
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
        float probability;
        float minimum_allowable_distance;

        ////////////////////////////////////////////////////////////
        float xC, yC, radius, y;
        ////////////////////////////////////////////////////////////

        for (k = index_y; k < K; k += stride_y) {
            for (i = index_x; i < N; i += stride_x) {
                p = &(X[D*i]);
                j = edgesIn[K*i+k];
                q = &(X[D*j]);
                // Initialize the probability to 1 and reduce it from
                // there
                probabilities[K*i+k] = 1;

                length_squared = 0;
                for(d = 0; d < D; d++) {
                    pq[d] = p[d] - q[d];
                    length_squared += pq[d]*pq[d];
                }
                // A point should not be connected to itself
                if(length_squared == 0) {
                    probabilities[K*i+k] = 0;
                    continue;
                }

                // for(n = 0; n < N; n++) {
                for(k2 = 0; k2 < 2*K; k2++) {
                    n = (k2 < K) ? edgesIn[K*i+k2] : edgesIn[K*j+(k2-K)];
                    r = &(X[D*n]);

                    // t is the parameterization of the projection of pr
                    // onto pq. In layman's terms, this is the length of
                    // the shadow pr casts onto pq
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

                        ////////////////////////////////////////////////
                        // ported from python function, can possibly be
                        // improved in terms of performance
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

                        probability = logistic_function(sqrt(squared_distance_to_edge), minimum_allowable_distance, steepness);

                        if(probability < probabilities[K*i+k]) {
                            probabilities[K*i+k] = probability;
                        }
                    }
                }
            }
        }
    }

    __global__
    void probability_relaxed_d(float *X,
                               int *edgesIn,
                               int N,
                               int D,
                               int K,
                               float lp,
                               float beta,
                               float steepness,
                               float *probabilities) {
        int index_x = blockIdx.x * blockDim.x + threadIdx.x;
        int stride_x = blockDim.x * gridDim.x;

        // We should use a 1D structure for this since we need to guarantee
        // that other points have already been processed
        // int index_y = blockIdx.y * blockDim.y + threadIdx.y;
        // int stride_y = blockDim.y * gridDim.y;

        float *p, *q, *r;

        float pq[10] = {};
        float pr[10] = {};

        int i, j, k, k2, d, n;
        float t;

        float length_squared;
        float squared_distance_to_edge;
        float probability;
        float minimum_allowable_distance;

        ////////////////////////////////////////////////////////////
        float xC, yC, radius, y;
        ////////////////////////////////////////////////////////////

        for (i = index_x; i < N; i += stride_x) {
            for (k = 0; k < K; k++) {
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
                    probabilities[K*i+k] = 0;
                    continue;
                }

                // This loop presumes that all nearer neighbors have
                // already been processed
                for(k2 = 0; k2 < k; k2++) {
                    n = edgesIn[K*i+k2];
                    if (n == -1 || probabilities[K*i+k2] < 1e-6 ){
                        continue;
                    }
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

                        probability = logistic_function(sqrt(squared_distance_to_edge), minimum_allowable_distance, steepness);

                        if(probability < probabilities[K*i+k]) {
                            probabilities[K*i+k] = probability;
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

    void prune_discrete(float *X, int *edges, int *indices, int N, int D, int M, int K,
                        float *erTemplate, int steps, bool relaxed, float beta,
                        float p, int count) {
        float *x_d;
        int *edgesIn_d;
        int *edgesOut_d;
        float *erTemplate_d;

        if (count < 0) {
            count = N;
        }

        cudaErrchk(cudaMallocManaged(&edgesIn_d, M*K*sizeof(int)));
        memcpy(edgesIn_d, edges, M*K*sizeof(int));

        cudaErrchk(cudaMallocManaged(&edgesOut_d, count*K*sizeof(int)));
        memcpy(edgesOut_d, edges, count*K*sizeof(int));

        if (indices != NULL) {
            int *map_d;
            int i;
            int max_idx = 0;
            for(i = 0; i < count; i++) {
                max_idx = indices[i] > max_idx ? indices[i] : max_idx;
            }

            cudaMallocManaged(&map_d, max_idx*sizeof(int));
            for(i = 0; i < count; i++) {
                map_d[indices[i]] = i;
            }
            map_indices(edgesIn_d, map_d, M, K);
            map_indices(edgesOut_d, map_d, M, K);
            cudaFree(map_d);
        }

        cudaErrchk(cudaMallocManaged(&x_d, N*D*sizeof(float)));
        memcpy(x_d, X, N*D*sizeof(float));


        cudaErrchk(cudaMallocManaged(&erTemplate_d, steps*sizeof(float)));
        if (erTemplate != NULL) {
            memcpy(erTemplate_d, erTemplate, (steps)*sizeof(float));
        }
        else {
            float temp_erTemplate[steps];
            create_template(temp_erTemplate, beta, p, steps);
            memcpy(erTemplate_d, temp_erTemplate, (steps)*sizeof(float));
        }

        if (relaxed) {
            prune_discrete_relaxed_d<<<grid_size_1D, block_size_1D>>>(x_d,
                                                                      edgesIn_d,
                                                                      count,
                                                                      D,
                                                                      K,
                                                                      erTemplate_d,
                                                                      steps,
                                                                      edgesOut_d);
        }
        else {
            prune_discrete_d<<<grid_size, block_size>>>(x_d,
                                                        edgesIn_d,
                                                        count,
                                                        D,
                                                        K,
                                                        erTemplate_d,
                                                        steps,
                                                        edgesOut_d);
        }
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            printf("Error: %s\n", cudaGetErrorString(err));
        cudaDeviceSynchronize();

        if (indices != NULL) {
            unmap_indices(edgesOut_d, indices, M, K, count);
        }

        memcpy(edges, edgesOut_d, count*K*sizeof(int));

        cudaFree(x_d);
        cudaFree(edgesIn_d);
        cudaFree(edgesOut_d);
        cudaFree(erTemplate_d);
    }

    void prune(float *X, int *edges, int *indices, int N, int D, int M, int K,
               bool relaxed, float beta, float lp, int count) {
        float *x_d;
        int *edgesIn_d;
        int *edgesOut_d;

        if (count < 0) {
            count = N;
        }

        auto start = std::chrono::high_resolution_clock::now();
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "\t" << "Memory allocation for edges (" << get_available_device_memory() - M*K*sizeof(int) - count*K*sizeof(int) << "): " << std::flush;

        cudaMallocManaged(&edgesIn_d, M*K*sizeof(int));
        memcpy(edgesIn_d, edges, M*K*sizeof(int));

        cudaMallocManaged(&edgesOut_d, count*K*sizeof(int));
        memcpy(edgesOut_d, edges, count*K*sizeof(int));

        end = std::chrono::high_resolution_clock::now();
	    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() / 1000. << " s" << std::endl;
        start = std::chrono::high_resolution_clock::now();
	    std::cout << "\t" << "Memory allocation for index map " << std::flush;

        if (indices != NULL) {
            int *map_d;
            int i;

            int max_index = 0;
            for(i = 0; i < N; i++) {
                if (indices[i] > max_index) {
                    max_index = indices[i];
                }
            }

            std::cout << "(" << get_available_device_memory() - max_index*sizeof(int) << "): " << std::flush;
            cudaMallocManaged(&map_d, max_index*sizeof(int));
            for(i = 0; i < N; i++) {
                map_d[indices[i]] = i;
            }

            end = std::chrono::high_resolution_clock::now();
	        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() / 1000. << " s" << std::endl;
            start = std::chrono::high_resolution_clock::now();
	        std::cout << "\t" << "Mapping indices for edgesIn: " << std::flush;

            map_indices(edgesIn_d, map_d, M, K);

            end = std::chrono::high_resolution_clock::now();
	        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() / 1000. << " s" << std::endl;
            start = std::chrono::high_resolution_clock::now();
	        std::cout << "\t" << "Mapping indices for edgesOut: " << std::flush;

            map_indices(edgesOut_d, map_d, count, K);

            end = std::chrono::high_resolution_clock::now();
	        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() / 1000. << " s" << std::endl;
            start = std::chrono::high_resolution_clock::now();
	        std::cout << "\t" << "Freeing map index: " << std::flush;

            cudaFree(map_d);
        }

        end = std::chrono::high_resolution_clock::now();
	    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() / 1000. << " s" << std::endl;
        start = std::chrono::high_resolution_clock::now();
	    std::cout << "\t" << "Memory allocation for X (" << get_available_device_memory() - N*D*sizeof(float) << "): " << std::flush;

        cudaMallocManaged(&x_d, N*D*sizeof(float));
        memcpy(x_d, X, N*D*sizeof(float));

        end = std::chrono::high_resolution_clock::now();
	    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() / 1000. << " s" << std::endl;
        start = std::chrono::high_resolution_clock::now();
	    std::cout << "\t" << "Device call: " << std::flush;

        if (relaxed) {
            prune_relaxed_d<<<grid_size_1D, block_size_1D>>>(x_d, edgesIn_d, count, D, K, lp, beta, edgesOut_d);
        }
        else {
            prune_d<<<grid_size, block_size>>>(x_d, edgesIn_d, count, D, K, lp, beta, edgesOut_d);
        }

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            printf("Error: %s\n", cudaGetErrorString(err));
        cudaDeviceSynchronize();

        end = std::chrono::high_resolution_clock::now();
	    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() / 1000. << " s" << std::endl;
        start = std::chrono::high_resolution_clock::now();

        if (indices != NULL) {
            unmap_indices(edgesOut_d, indices, count, K, N);
        }

        end = std::chrono::high_resolution_clock::now();
        std::cout << "\t" << "Unmap indices: " << std::flush;
	    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() / 1000. << " s" << std::endl;
        start = std::chrono::high_resolution_clock::now();
	    std::cout << "\t" << "Copying back: " << std::flush;

        memcpy(edges, edgesOut_d, count*K*sizeof(int));

        end = std::chrono::high_resolution_clock::now();
	    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() / 1000. << " s" << std::endl;
        start = std::chrono::high_resolution_clock::now();
	    std::cout << "\t" << "Free remaining memory: " << std::flush;

        cudaFree(x_d);
        cudaFree(edgesIn_d);
        cudaFree(edgesOut_d);

	    end = std::chrono::high_resolution_clock::now();
	    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() / 1000. << " s" << std::endl;
    }

    __global__
    void set_value_d(int *X, int value, int N) {
        int index_x = blockIdx.x * blockDim.x + threadIdx.x;
        int stride_x = blockDim.x * gridDim.x;

        for (int i = index_x; i < N; i += stride_x) {
            X[i] = value;
        }
    }

    __global__
    void prune_yao_d(float *X, int *edgesIn, float *bisectors, int *vacancies,
                     int N, int D, int K, int numSectors, int *edgesOut) {
        int index_x = blockIdx.x * blockDim.x + threadIdx.x;
        int stride_x = blockDim.x * gridDim.x;

        // int index_y = blockIdx.y * blockDim.y + threadIdx.y;
        // int stride_y = blockDim.y * gridDim.y;

        int i, j, k, d;

        float *p, *q;
        float pq[10] = {};
        float length_squared;

        for (i = index_x; i < N; i += stride_x) {
            for (k = 0; k < K; k++) {
                p = &(X[D*i]);
                j = edgesIn[K*i+k];
                q = &(X[D*j]);

                length_squared = 0;
                for(d = 0; d < D; d++) {
                    pq[d] = q[d] - p[d];
                    length_squared += pq[d]*pq[d];
                }
                // A point should not be connected to itself
                if(length_squared == 0) {
                    edgesOut[K*i+k] = -1;
                    continue;
                }

                // pq dot bisectors
                int representative_sector = -1;
                float max_projection = 0;
                for( j = 0; j < numSectors; j++) {
                    float projection = 0;
                    for (d = 0; d < D; d++) {
                        projection += pq[d]*bisectors[j*D+d];
                    }
                    if (projection > max_projection || representative_sector < 0) {
                        max_projection = projection;
                        representative_sector = j;
                    }
                }

                if (vacancies[i*numSectors + representative_sector] > 0) {
                    vacancies[i*numSectors + representative_sector]--;
                }
                else {
                    edgesOut[K*i+k] = -1;
                }

            }
        }
    }

    void prune_yao(float *X, float *bisectors, int *edges, int *indices, int N, int D, int M, int K,
               int numSectors, int numPointsPerSector, int count) {
        float *x_d;
        float *bisectors_d;
        int *edgesIn_d;
        int *edgesOut_d;
        int *vacancies_d;

        if (count < 0) {
            count = N;
        }

        auto start = std::chrono::high_resolution_clock::now();
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "\t" << "Memory allocation for edges (" << get_available_device_memory() - M*K*sizeof(int) - count*K*sizeof(int) << "): " << std::flush;

        cudaMallocManaged(&edgesIn_d, M*K*sizeof(int));
        memcpy(edgesIn_d, edges, M*K*sizeof(int));

        cudaMallocManaged(&edgesOut_d, count*K*sizeof(int));
        memcpy(edgesOut_d, edges, count*K*sizeof(int));

        cudaMallocManaged(&bisectors_d, numSectors*D*sizeof(int));
        memcpy(bisectors_d, bisectors, numSectors*D*sizeof(int));


        end = std::chrono::high_resolution_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() / 1000. << " s" << std::endl;

        start = std::chrono::high_resolution_clock::now();
        std::cout << "\t" << "Memory allocation for vacancy array: " << std::flush;

        cudaMallocManaged(&vacancies_d, count*numSectors*sizeof(int));
        set_value_d<<<grid_size_1D, block_size_1D>>>(vacancies_d, numPointsPerSector, count*numSectors);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            printf("Error: %s\n", cudaGetErrorString(err));
        cudaDeviceSynchronize();

        end = std::chrono::high_resolution_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() / 1000. << " s" << std::endl;
        start = std::chrono::high_resolution_clock::now();

        start = std::chrono::high_resolution_clock::now();
	    std::cout << "\t" << "Memory allocation for index map " << std::flush;

        if (indices != NULL) {
            int *map_d;
            int i;

            int max_index = 0;
            for(i = 0; i < N; i++) {
                if (indices[i] > max_index) {
                    max_index = indices[i];
                }
            }

            std::cout << "(" << get_available_device_memory() - max_index*sizeof(int) << "): " << std::flush;
            cudaMallocManaged(&map_d, max_index*sizeof(int));
            for(i = 0; i < N; i++) {
                map_d[indices[i]] = i;
            }

            end = std::chrono::high_resolution_clock::now();
	        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() / 1000. << " s" << std::endl;
            start = std::chrono::high_resolution_clock::now();
	        std::cout << "\t" << "Mapping indices for edgesIn: " << std::flush;

            map_indices(edgesIn_d, map_d, M, K);

            end = std::chrono::high_resolution_clock::now();
	        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() / 1000. << " s" << std::endl;
            start = std::chrono::high_resolution_clock::now();
	        std::cout << "\t" << "Mapping indices for edgesOut: " << std::flush;

            map_indices(edgesOut_d, map_d, count, K);

            end = std::chrono::high_resolution_clock::now();
	        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() / 1000. << " s" << std::endl;
            start = std::chrono::high_resolution_clock::now();
	        std::cout << "\t" << "Freeing map index: " << std::flush;

            cudaFree(map_d);
        }

        end = std::chrono::high_resolution_clock::now();
	    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() / 1000. << " s" << std::endl;
        start = std::chrono::high_resolution_clock::now();
	    std::cout << "\t" << "Memory allocation for X (" << get_available_device_memory() - N*D*sizeof(float) << "): " << std::flush;

        cudaMallocManaged(&x_d, N*D*sizeof(float));
        memcpy(x_d, X, N*D*sizeof(float));

        end = std::chrono::high_resolution_clock::now();
	    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() / 1000. << " s" << std::endl;
        start = std::chrono::high_resolution_clock::now();
	    std::cout << "\t" << "Device call: " << std::flush;

        prune_yao_d<<<grid_size_1D, block_size_1D>>>(x_d, edgesIn_d, bisectors_d, vacancies_d, count, D, K, numSectors, edgesOut_d);

        err = cudaGetLastError();
        if (err != cudaSuccess)
            printf("Error: %s\n", cudaGetErrorString(err));
        cudaDeviceSynchronize();

        end = std::chrono::high_resolution_clock::now();
	    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() / 1000. << " s" << std::endl;
        start = std::chrono::high_resolution_clock::now();

        if (indices != NULL) {
            unmap_indices(edgesOut_d, indices, count, K, N);
        }

        end = std::chrono::high_resolution_clock::now();
        std::cout << "\t" << "Unmap indices: " << std::flush;
	    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() / 1000. << " s" << std::endl;
        start = std::chrono::high_resolution_clock::now();
	    std::cout << "\t" << "Copying back: " << std::flush;

        memcpy(edges, edgesOut_d, count*K*sizeof(int));

        end = std::chrono::high_resolution_clock::now();
	    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() / 1000. << " s" << std::endl;
        start = std::chrono::high_resolution_clock::now();
	    std::cout << "\t" << "Free remaining memory: " << std::flush;

        cudaFree(x_d);
        cudaFree(edgesIn_d);
        cudaFree(edgesOut_d);
        cudaFree(bisectors_d);
        cudaFree(vacancies_d);

	    end = std::chrono::high_resolution_clock::now();
	    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() / 1000. << " s" << std::endl;
    }

    void associate_probability(float *X,
                               int *edges,
                               float *probabilities,
                               int *indices,
                               int N,
                               int D,
                               int M,
                               int K,
                               float steepness,
                               bool relaxed,
                               float beta,
                               float lp,
                               int count) {
        float *x_d;
        int *edgesIn_d;
        float *probabilities_d;

        if (count < 0) {
            count = N;
        }

        cudaMallocManaged(&edgesIn_d, M*K*sizeof(int));
        memcpy(edgesIn_d, edges, M*K*sizeof(int));

        cudaMallocManaged(&probabilities_d, count*K*sizeof(float));
        // We don't care what probabilities_d holds initially, we will
        // overwrite it.

        if (indices != NULL) {
            int *map_d;
            int i;

            cudaMallocManaged(&map_d, N*sizeof(int));
            for(i = 0; i < N; i++) {
                map_d[indices[i]] = i;
            }
            map_indices(edgesIn_d, map_d, M, K);
            cudaFree(map_d);
        }

        cudaMallocManaged(&x_d, N*D*sizeof(float));
        memcpy(x_d, X, N*D*sizeof(float));

        if (relaxed) {
            probability_relaxed_d<<<grid_size, block_size>>>(x_d,
                                                             edgesIn_d,
                                                             count,
                                                             D,
                                                             K,
                                                             lp,
                                                             beta,
                                                             steepness,
                                                             probabilities_d);
        }
        else {
            probability_d<<<grid_size, block_size>>>(x_d,
                                                     edgesIn_d,
                                                     count,
                                                     D,
                                                     K,
                                                     lp,
                                                     beta,
                                                     steepness,
                                                     probabilities_d);
        }
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            printf("Error: %s\n", cudaGetErrorString(err));
        cudaDeviceSynchronize();

        memcpy(probabilities, probabilities_d, count*K*sizeof(float));

        cudaFree(x_d);
        cudaFree(edgesIn_d);
        cudaFree(probabilities_d);
    }

    void print_cuda_info() {
        struct cudaDeviceProp properties;
        cudaGetDeviceProperties(&properties, 0);
        fprintf(stderr, "using %d multiprocessors\n", properties.multiProcessorCount);
        fprintf(stderr, "max threads per processor: %d\n", properties.maxThreadsPerMultiProcessor);
        fprintf(stderr, "Grid Size: %dx%d\n", grid_size.x, grid_size.y);
        fprintf(stderr, "Block Size: %dx%d\n", block_size.x, block_size.y);
    }

    size_t get_available_device_memory() {
        size_t free_memory;
        size_t total_memory;
        cudaErrchk(cudaMemGetInfo(&free_memory, &total_memory));

        return free_memory;
    }
}
