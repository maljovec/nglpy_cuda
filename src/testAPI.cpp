#include "Graph.h"
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <sys/time.h>
#include <unistd.h>
#include <cstdlib>
#include <fstream>

typedef struct timeval timestamp;
static inline float operator-(const timestamp &t1, const timestamp &t2)
{
    return (float)(t1.tv_sec - t2.tv_sec) +
           1.0e-6f * (t1.tv_usec - t2.tv_usec);
}
static inline timestamp now()
{
    timestamp t;
    gettimeofday(&t, NULL);
    return t;
}

class CommandLine
{
    std::vector<std::string> argnames;
    std::map<std::string, std::string> args;
    std::map<std::string, std::string> descriptions;
    std::vector<std::string> requiredArgs;

  public:
    CommandLine()
    {
    }
    void addArgument(std::string cmd, std::string defaultValue, std::string description = "", bool required = false)
    {
        argnames.push_back(cmd);
        if (description != "")
        {
            descriptions[cmd] = description;
        }
        if (required && find(requiredArgs.begin(), requiredArgs.end(), cmd) == requiredArgs.end())
        {
            requiredArgs.push_back(cmd);
        }
        setArgument(cmd, defaultValue);
    }
    void setArgument(std::string cmd, std::string value)
    {
        args[cmd] = value;
    }
    bool processArgs(int argc, char *argv[])
    {
        std::map<std::string, int> check;
        for (unsigned int k = 0; k < requiredArgs.size(); k++)
        {
            check[requiredArgs[k]] = 0;
        }
        for (int i = 1; i < argc; i += 2)
        {
            if (i + 1 < argc)
            {
                std::string arg = std::string(argv[i]);
                std::string val = std::string(argv[i + 1]);
                setArgument(arg, val);
                if (check.find(arg) != check.end())
                {
                    check[arg] = check[arg] + 1;
                }
            }
        }
        for (std::map<std::string, int>::const_iterator it = check.begin(); it != check.end(); it++)
        {
            int n = it->second;
            if (n == 0)
            {
                return false;
            }
        }
        return true;
    }
    void showUsage()
    {
        for (unsigned int k = 0; k < argnames.size(); k++)
        {
            fprintf(stderr, "\t%s\t\t%s (%s)\n", argnames[k].c_str(), descriptions[argnames[k]].c_str(), args[argnames[k]].c_str());
        }
    }
    float getArgFloat(std::string arg)
    {
        std::string val = args[arg];
        return atof(val.c_str());
    }
    int getArgInt(std::string arg)
    {
        std::string val = args[arg];
        return atoi(val.c_str());
    }
    std::string getArgString(std::string arg)
    {
        std::string val = args[arg];
        return val;
    }
};

int main(int argc, char **argv)
{
    timestamp t1 = now();
    timestamp t2;
    CommandLine cl;
    cl.addArgument("-d", "2", "Number of dimensions", true);
    cl.addArgument("-c", "1000000", "Number of points", true);
    cl.addArgument("-k", "-1", "K max", false);
    cl.addArgument("-b", "1.0", "Beta", false);
    cl.addArgument("-p", "2.0", "Lp-norm", false);
    cl.addArgument("-r", "0", "Relaxed", false);
    cl.addArgument("-s", "-1", "# of Discretization Steps. Use -1 to disallow discretization.", false);
    bool hasArguments = cl.processArgs(argc, argv);
    if (!hasArguments)
    {
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
    bool relaxed = cl.getArgInt("-r") > 0;
    float beta = cl.getArgFloat("-b");
    float lp = cl.getArgFloat("-p");

    int i, d, k;

    t2 = now();
    fprintf(stderr, "Setup (%f s)\n", t2 - t1);
    t1 = now();

    float *x = new float[N * D];
    for (i = 0; i < N; i++)
    {
        for (d = 0; d < D; d++)
        {
            x[i * D + d] = (float)rand() / RAND_MAX;
        }
    }

    auto myFile = std::fstream("points.bin", std::ios::out | std::ios::binary);
    myFile.write((char *)x, N * D * sizeof(float));
    myFile.close();

    t2 = now();
    fprintf(stderr, "Generating Points (%f s)\n", t2 - t1);
    t1 = now();

    Graph g(NULL, K, relaxed, beta, lp, discrete);
    g.build(x, N, D);

    t2 = now();
    std::cerr << "Generating Graph " << t2 - t1 << " s" << std::endl;
    t1 = now();

    Edge e = g.next();
    while (e.indices[0] >= 0 && e.indices[0] < N)
    {
        std::cout << e.indices[0] << " " << e.indices[1] << " " << e.distance << std::endl;
        e = g.next();
    }

    t2 = now();
    std::cerr << "Output " << t2 - t1 << " s" << std::endl;
    t1 = now();

    // Free memory
    delete[] x;

    t2 = now();
    std::cerr << "Clean-up " << t2 - t1 << " s" << std::endl;

    return 0;
}