#ifndef UTILS_CUH
#define UTILS_CUH

#include <cstdio>
#include <iostream>
#include <chrono>

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
    dim3 block_size_1D(1024);
    dim3 grid_size_1D(16);

    __global__
    void unmap_indices_d(int *matrix, int *map, int M, int N) {
        int index_x = blockIdx.x * blockDim.x + threadIdx.x;
        int stride_x = blockDim.x * gridDim.x;

        int index_y = blockIdx.y * blockDim.y + threadIdx.y;
        int stride_y = blockDim.y * gridDim.y;

        int row, col;
        for (row = index_y; row < M; row += stride_y) {
            for (col = index_x; col < N; col += stride_x) {
                if(matrix[row*N+col] != -1) {
                    matrix[row*N+col] = map[matrix[row*N+col]];
                }
            }
        }
    }

    void map_indices(int *matrix_d, int *map_d, int M, int N) {
        unmap_indices_d<<<grid_size, block_size>>>(matrix_d, map_d, M, N);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            printf("Error: %s\n", cudaGetErrorString(err));
        cudaDeviceSynchronize();
    }

    void unmap_indices(int *matrix_d, int *map, int M, int N, int K) {
        int *map_d;

        auto start = std::chrono::high_resolution_clock::now();
        std::cout << "\n\t\tAllocating map_d (" << get_available_device_memory() - K*sizeof(int) << "): " << std::flush;

        cudaMallocManaged(&map_d, K*sizeof(int));
        memcpy(map_d, map, K*sizeof(int));

        auto end = std::chrono::high_resolution_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() / 1000. << " s" << std::endl;
        std::cout << "\t" << "\tDevice call: " << std::flush;
        start = std::chrono::high_resolution_clock::now();

        unmap_indices_d<<<grid_size, block_size>>>(matrix_d, map_d, M, N);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            printf("Error: %s\n", cudaGetErrorString(err));
        cudaDeviceSynchronize();

        end = std::chrono::high_resolution_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() / 1000. << " s" << std::endl;
        std::cout << "\t\tMemory free: " << std::flush;
        start = std::chrono::high_resolution_clock::now();

        cudaFree(map_d);

        end = std::chrono::high_resolution_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() / 1000. << " s" << std::endl;
    }
}

#endif //UTILS_CUH