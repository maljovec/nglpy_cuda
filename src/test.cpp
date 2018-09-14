#include "ngl_cuda.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <sys/time.h>
#include <unistd.h>
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


int main(int argc, char **argv)
{
  timestamp t1 = now();
  timestamp t2;
  CommandLine cl;
  cl.addArgument("-i", "input", "Input points", true);
  cl.addArgument("-d", "2", "Number of dimensions", true);
  cl.addArgument("-c", "1000000", "Number of points", true);
  cl.addArgument("-n", "neighbor", "Neighbor edge file", true);
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

  std::string pointFile = cl.getArgString("-i");
  int D = cl.getArgInt("-d");
  int N = cl.getArgInt("-c");
  int K = cl.getArgInt("-k");
  int steps = cl.getArgInt("-s");
  bool discrete = steps > 0;

  float beta = cl.getArgFloat("-b");
  float lp = cl.getArgFloat("-p");

  // Load data set and edges from files
  float *x;
  int *edgesOut;
  float *probabilities;
  float *referenceShape;

  int i, d, k;

  std::string outputFilename;

  x = new float[N*D];
  edgesOut = new int[N*K];
  probabilities = new float[N*K];
  if(discrete) {
    referenceShape = new float[steps+1];
  }

  t2 = now();
  std::cerr << "Setup and Memory Allocation " << t2-t1 << " s" << std::endl;
  t1 = now();

  std::ifstream file1( pointFile.c_str() );

  i = 0;
  d = 0;
  std::string line;

  while ( std::getline(file1, line) )
  {
    std::istringstream iss(line);
    for (d = 0; d < D; d++) {
      iss >> x[i*D+d];
    }
    i++;
  }
  file1.close();
  t2 = now();
  std::cerr << "Reading Points " << t2-t1 << " s" << std::endl;
  t1 = now();

  std::string edgeFile = cl.getArgString("-n");

  std::ifstream file2 ( edgeFile.c_str() );
  i = 0;
  while ( std::getline(file2, line) )
  {
      std::istringstream iss(line);
      for (k = 0; k < K; k++) {
          iss >> edgesOut[i*K+k];
      }
      i++;
  }
  file2.close();

  t2 = now();
  std::cerr << "Reading Graph " << t2-t1 << " s" << std::endl;
  t1 = now();

  if(discrete) {
      nglcu::create_template(referenceShape, 1, 2, steps);
      nglcu::prune_discrete(N, D, K, steps, referenceShape, x, edgesOut);
  }
  else {
      nglcu::prune(N, D, K, lp, beta, x, edgesOut);
  }

  nglcu::associate_probability(N, D, K, lp, beta, x, edgesOut, probabilities);

  t2 = now();
  std::cerr << "GPU execution " << t2-t1 << " s" << std::endl;
  t1 = now();

  for(i = 0; i < N; i++) {
    for(k = 0; k < K; k++) {
        if (k > 0) {
            std::cout << " ";
        }
        std::cout << edgesOut[i*K+k];
    }
    std::cout << std::endl;
  }

  std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
  std::cout << "     Probabilities" << std::endl;
  for(i = 0; i < N; i++) {
    for(k = 0; k < K; k++) {
        if (k > 0) {
            std::cout << " ";
        }
        std::cout << probabilities[i*K+k];
    }
    std::cout << std::endl;
  }

  t2 = now();
  std::cerr << "Output " << t2-t1 << " s" << std::endl;
  t1 = now();

  // Free memory
  delete [] x;
  delete [] edgesOut;
  delete [] probabilities;
  if(discrete) {
      delete [] referenceShape;
  }

  t2 = now();
  std::cerr << "Clean-up " << t2-t1 << " s" << std::endl;

  return 0;
}
