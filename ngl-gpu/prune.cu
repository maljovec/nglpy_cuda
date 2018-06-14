#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <ANN/ANN.h>

# include <sys/time.h>
# include <unistd.h>
  typedef struct timeval timestamp;
  static inline float operator - (const timestamp &t1, const timestamp &t2)
  {
	return (float)(t1.tv_sec  - t2.tv_sec) +
	       1.0e-6f*(t1.tv_usec - t2.tv_usec);
  }
  static inline timestamp now()
  {
	timestamp t;
	gettimeofday(&t, NULL);
	return t;
  }

class CommandLine {
	std::vector<std::string> argnames;
    std::map<std::string, std::string> args;
	std::map<std::string, std::string> descriptions;
	std::vector<std::string> requiredArgs;
  public:
    CommandLine() {
	}
	void addArgument(std::string cmd, std::string defaultValue, std::string description="", bool required = false) {
		argnames.push_back(cmd);
        if(description!="") {
            descriptions[cmd] = description;
        }
		if(required && find(requiredArgs.begin(), requiredArgs.end(), cmd)==requiredArgs.end()) {
			requiredArgs.push_back(cmd);
		}
		setArgument(cmd, defaultValue);
    }
	void setArgument(std::string cmd, std::string value) {
		args[cmd] = value;
	}
	bool processArgs(int argc, char *argv[]) {
		std::map<std::string, int> check;
		for(unsigned int k = 0;k<requiredArgs.size();k++) {
			check[requiredArgs[k]] = 0;
		}
		for(int i=1; i<argc; i+=2) {
			if(i+1<argc) {
				std::string arg = std::string(argv[i]);
				std::string val = std::string(argv[i+1]);
				setArgument(arg, val);
				if(check.find(arg)!=check.end()) {
					check[arg] = check[arg] + 1;
				}
			}
		}
		for(std::map<std::string,int>::const_iterator it = check.begin(); it!=check.end();it++) {
			int n = it->second;
            if(n==0) {
                return false;
            }
		}
		return true;
	}
	void showUsage() {
		for(unsigned int k = 0; k<argnames.size();k++) {
			fprintf(stderr, "\t%s\t\t%s (%s)\n", argnames[k].c_str(), descriptions[argnames[k]].c_str(), args[argnames[k]].c_str());
		}
	}
	float getArgFloat(std::string arg) {
		std::string val = args[arg];
		return atof(val.c_str());
	}													 
    int getArgInt(std::string arg) {
        std::string val = args[arg];
		return atoi(val.c_str());
	}
    std::string getArgString(std::string arg) {
		std::string val = args[arg];
		return val;
	}
};

__global__
void prune_discrete(const int N, const int D, const int K, const int steps,
                    float *X, int *edgesIn, int *edgesOut, float *erTemplate)
{
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
            for(d = 0; d < D; d++){
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
                for(d = 0; d < D; d++){
                    pr[d] = p[d] - r[d];
                    t += pr[d]*pq[d];
                }

                t /= length_squared;
                lookup = __float2int_rd(abs(steps * (2 * t - 1))+0.5);
                if (lookup >= 0 && lookup <= steps) {
                    squared_distance_to_edge = 0;
                    for(d = 0; d < D; d++){
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
void prune(const int N, const int D, const int K, float *X, int *edgesIn,
           int *edgesOut, float lp, float beta)
{
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
            for(d = 0; d < D; d++){
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
                for(d = 0; d < D; d++){
                    pr[d] = p[d] - r[d];
                    t += pr[d]*pq[d];
                }

                t /= length_squared;

                if (t > 0 && t < 1) {
                    squared_distance_to_edge = 0;
                    for(d = 0; d < D; d++){
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

float minDistanceFromEdge(float t, float beta, float p) {
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

void createTemplate(float * data, float beta=1, int p=2, int steps=100) {
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
            data[i] = minDistanceFromEdge(float(i)/steps, beta, p);
        }
    }
}


int main(int argc, char **argv)
{
  timestamp t1 = now();
  timestamp t2;
  CommandLine cl;
  cl.addArgument("-i", "input", "Input points", true);
  cl.addArgument("-d", "2", "Number of dimensions", true);
  cl.addArgument("-n", "1000000", "Number of points", true);
  cl.addArgument("-k", "-1", "K max", false);
  cl.addArgument("-b", "1.0", "Beta", false);
  cl.addArgument("-p", "2.0", "Lp-norm", false);
  cl.addArgument("-s", "-1", "# of Discretization Steps. Use -1 to disallow discretization.", false);
  bool hasArguments = cl.processArgs(argc, argv);
  if(!hasArguments) {
    fprintf(stderr, "Missing arguments\n");
    fprintf(stderr, "Usage:\n\n");
    cl.showUsage();
    exit(1);
  }

  struct cudaDeviceProp properties;
  cudaGetDeviceProperties(&properties, 0);
  std::cerr << "using " << properties.multiProcessorCount << " multiprocessors"
            << std::endl 
            << "max threads per processor: " 
            << properties.maxThreadsPerMultiProcessor << std::endl;

  std::string pointFile = cl.getArgString("-i");
  int D = cl.getArgInt("-d");
  int N = cl.getArgInt("-n");
  int K = cl.getArgInt("-k");
  int steps = cl.getArgInt("-s");
  bool discrete = steps > 0;

  float beta = cl.getArgFloat("-b");
  float lp = cl.getArgFloat("-p");

  // Load data set and edges from files
  // TODO
  float *x;
  int *edgesIn;
  int *edgesOut;
  float *referenceShape;
  dim3 blockSize(32, 32);
  dim3 gridSize(4, 4);

  std::cerr << "Grid Size: " << gridSize.x << "x" << gridSize.y << std::endl
            << "Block Size: " << blockSize.x << "x" << blockSize.y << std::endl;
  int i, d, k;

  std::string outputFilename;

  cudaMallocManaged(&x, N*D*sizeof(float));
  cudaMallocManaged(&edgesIn, N*K*sizeof(int));
  cudaMallocManaged(&edgesOut, N*K*sizeof(int));
  if(discrete) {
    cudaMallocManaged(&referenceShape, (steps+1)*sizeof(float));
  }

  t2 = now();
  std::cerr << "Setup and Memory Allocation " << t2-t1 << " s" << std::endl;
  t1 = now();

  std::ifstream file1( pointFile );
  
  i = 0;
  d = 0;
  std::string line;
  
  while ( std::getline(file1, line) )
  {
    std::istringstream iss(line);
    for (d = 0; d < D; d++) {
      iss >> x[i*D+d];
    //   dataPts[i][d] = x[i*D+d];
    }
    i++;
  }
  file1.close();
  t2 = now();
  std::cerr << "Reading Data " << t2-t1 << " s" << std::endl;
  t1 = now();
  
  std::stringstream ss;

  ss << "../data/misc/knn_" << D << "D_" << N << ".txt";
  std::string edgeFile = ss.str();

  std::ifstream file2 ( edgeFile );
  i = 0;
  while ( std::getline(file2, line) )
  {
      std::istringstream iss(line);
      for (k = 0; k < K; k++) {
          iss >> edgesIn[i*K+k];
          edgesOut[i*K+k] = edgesIn[i*K+k];
      }
      i++;
  }
  file2.close();

//   ANNpointArray dataPts;
//   ANNpoint queryPt;
//   ANNidxArray nnIdx;
//   ANNdistArray dists;
//   ANNkd_tree* kdTree;

//   dataPts = annAllocPts(N, D);
//   kdTree = new ANNkd_tree(dataPts, N, D);
//   queryPt = annAllocPt(D);
//   nnIdx = new ANNidx[K];
//   dists = new ANNdist[K];
//   for(i = 0; i < N; i++) {
//     for(d = 0; d < D; d++) {
//         queryPt[d] = x[i*D+d];
//     }
//     kdTree->annkSearch(queryPt, K, nnIdx, dists, 0.f);
//     for(int k=0;k<K;k++) {
//         edgesOut[i*K+k] = edgesIn[i*K+k] = nnIdx[k];
//     }
//   }

//   annDeallocPts(dataPts);
//   annDeallocPt(queryPt);
//   delete nnIdx;
//   delete dists;
//   delete kdTree;cd ..
  t2 = now();
  std::cerr << "ANN computation " << t2-t1 << " s" << std::endl;
  t1 = now();

  if(discrete) {
    createTemplate(referenceShape, 1, 2, steps);
    prune_discrete<<<gridSize, blockSize>>>(N, D, K, steps, x, edgesIn, edgesOut, referenceShape);
  }
  else {
    prune<<<gridSize, blockSize>>>(N, D, K, x, edgesIn, edgesOut, lp, beta);
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) 
      printf("Error: %s\n", cudaGetErrorString(err));

  cudaDeviceSynchronize();
  t2 = now();
  std::cerr << "GPU execution " << t2-t1 << " s" << std::endl;
  t1 = now();

  for(i = 0; i < N; i++) {
    for(k = 0; k < K; k++) {
      if (edgesOut[i*K+k] != -1) {
        std::cout << i << " " << edgesOut[i*K+k] << std::endl;
      }
    }
  }

  // Free memory
  cudaFree(x);
  cudaFree(edgesIn);
  cudaFree(edgesOut);
  if(discrete) {
    cudaFree(referenceShape);
  }

  t2 = now();
  std::cerr << "Output and Clean-up " << t2-t1 << " s" << std::endl;

  return 0;
}