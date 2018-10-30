/*
 * Graph.h
 *
 *  Created on: Oct 22, 2018
 *      Author: maljovec
 */

#include "Graph.h"
#include "SearchIndex.h"
#include "ANNSearchIndex.h"
#include "ngl_cuda.h"
#include <cstdlib>
#include <vector>
#include <set>
#include <chrono>
#include <iostream>

Graph::Graph(SearchIndex *index,
             int maxNeighbors,
             bool relaxed,
             float beta,
             float p,
             int discreteSteps,
             int querySize)
    : mMaxNeighbors(maxNeighbors),
      mRelaxed(relaxed),
      mBeta(beta),
      mLp(p),
      mDiscreteSteps(discreteSteps),
      mQuerySize(querySize)
{
    if (index != NULL)
    {
        mSearchIndex = index;
    }
    else
    {
        mSearchIndex = new ANNSearchIndex(0);
        mSelfConstructedIndex = true;
    }
}

void Graph::build(float *X, int N, int D)
{
    mData = X;
    mCount = N;
    mDim = D;
    mSearchIndex->fit(mData, mCount, mDim);

    size_t availableGPUMemory = nglcu::get_available_device_memory();
    if (mQuerySize < 0)
    {
        // Because we are using f32 and i32:
        int bytesPerNumber = 4;
        int k = mMaxNeighbors;
        int worstCase;

        // Worst-case upper bound limit of the number of points
        // needed in a single query
        if (mRelaxed)
        {
            // For the relaxed algorithm we need n*D storage for the
            // point locations plus n*k for the representation of the
            // input edges plus another n*k for the output edges
            // We could potentially only need one set of edges for
            // this version
            worstCase = D + 2 * k;
        }
        else
        {
            // For the strict algorithm, if we are processing n
            // points at a time, we need the point locations of all
            // of their neigbhors, thus in the worst case, we need
            // n + n*k point locations and rows of the edge matrix.
            // the 2*k again represents the need for two versions of
            // the edge matrix. Here, we definitely need an input and
            // an output array
            worstCase = (D + 2 * k) * (k + 1);
        }

        // If we are using the discrete algorithm, remember we need
        // to add the template's storage as well to the GPU
        if (mDiscreteSteps > 0)
        {
            availableGPUMemory -= mDiscreteSteps * bytesPerNumber;
        }

        int divisor = bytesPerNumber * worstCase;
        mQuerySize = std::min((int)availableGPUMemory / divisor, mCount);
    }
    mChunked = mQuerySize < mCount;
    populate();
    // Load up the first edge
    while (mEdges[(mCurrentRow - mRowOffset) * mMaxNeighbors + mCurrentCol] == -1)
    {
        advanceIteration();
    }
}

void Graph::populate()
{
    if (mChunked)
    {
        mEdges = new int[mQuerySize * mMaxNeighbors];
        mDistances = new float[mQuerySize * mMaxNeighbors];
        populate_chunk(0);
    }
    else
    {
        mEdges = new int[mCount * mMaxNeighbors];
        mDistances = new float[mCount * mMaxNeighbors];
        populate_whole();
    }
}

void Graph::populate_chunk(int startIndex)
{
    mRowOffset = startIndex;
    int count = std::min(mCount - startIndex, mQuerySize);
    int edgeCount = count;
    int endIndex = startIndex + count;

    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "Initial KNN retrieval: " << std::flush;
    mSearchIndex->search(mRowOffset, count, mMaxNeighbors, mEdges, mDistances);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() / 1000. << " s" << std::endl;
    start = std::chrono::high_resolution_clock::now();
    std::cout << "Calculating additional indices: " << std::flush;

    std::set<int> additionalIndices;
    for (int i = 0; i < count; i++)
    {
        for (int k = 0; k < mMaxNeighbors; k++)
        {
            if (mEdges[i * mMaxNeighbors + k] < startIndex || mEdges[i * mMaxNeighbors + k] >= endIndex)
            {
                additionalIndices.insert(mEdges[i * mMaxNeighbors + k]);
            }
        }
    }

    end = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() / 1000. << " s" << std::endl;
    start = std::chrono::high_resolution_clock::now();
    std::cout << "Stacking indices: " << std::flush;

    std::vector<int> indices;
    for (int i = startIndex; i < endIndex; i++)
    {
        indices.push_back(i);
    }

    if (additionalIndices.size() > 0)
    {
        int extraCount = additionalIndices.size();
        for (auto it = additionalIndices.begin(); it != additionalIndices.end(); it++)
        {
            indices.push_back(*it);
        }

        end = std::chrono::high_resolution_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() / 1000. << " s" << std::endl;
        start = std::chrono::high_resolution_clock::now();

        if (!mRelaxed)
        {
            std::cout << "Secondary knn query: " << std::flush;
            int *extraEdges = new int[extraCount * mMaxNeighbors];
            int *extraIndices = indices.data() + count;
            mSearchIndex->search(extraIndices, extraCount, mMaxNeighbors, extraEdges, NULL);

            end = std::chrono::high_resolution_clock::now();
            std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() / 1000. << " s" << std::endl;
            start = std::chrono::high_resolution_clock::now();
            std::cout << "Stacking edges: " << std::flush;

            delete extraIndices;
            edgeCount = count + extraCount;
            int *allEdges = new int[edgeCount * mMaxNeighbors];
            for (int i = 0; i < count; i++)
            {
                for (int k = 0; k < mMaxNeighbors; k++)
                {
                    allEdges[i * mMaxNeighbors + k] = mEdges[i * mMaxNeighbors + k];
                }
            }
            std::set<int> uniqueIndices(indices.begin(), indices.end());
            additionalIndices.clear();
            for (int i = 0; i < extraCount; i++)
            {
                for (int k = 0; k < mMaxNeighbors; k++)
                {
                    allEdges[(i + count) * mMaxNeighbors + k] = extraEdges[i * mMaxNeighbors + k];
                    additionalIndices.insert(extraEdges[i * mMaxNeighbors + k]);
                }
            }
            delete mEdges;
            delete extraEdges;
            mEdges = allEdges;

            end = std::chrono::high_resolution_clock::now();
            std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() / 1000. << " s" << std::endl;
            start = std::chrono::high_resolution_clock::now();
            std::cout << "Stacking neighboring indices: " << std::flush;

            for (auto it = additionalIndices.begin(); it != additionalIndices.end(); it++)
            {
                indices.push_back(*it);
            }

            end = std::chrono::high_resolution_clock::now();
            std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() / 1000. << " s" << std::endl;
            start = std::chrono::high_resolution_clock::now();
        }
    }
    std::cout << "Subsetting X: " << std::flush;
    float *X = new float[indices.size() * mDim];
    for (int i = 0; i < indices.size(); i++)
    {
        for (int d = 0; d < mDim; d++)
        {
            X[i * mDim + d] = mData[indices[i] * mDim + d];
        }
    }

    end = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() / 1000. << " s" << std::endl;
    start = std::chrono::high_resolution_clock::now();

    if (mDiscreteSteps > 0)
    {
        nglcu::prune_discrete(X, mEdges, indices.data(), indices.size(), mDim, edgeCount,
                              mMaxNeighbors, NULL, mDiscreteSteps, mRelaxed, mBeta, mLp, count);
    }
    else
    {
        nglcu::prune(X, mEdges, indices.data(), indices.size(), mDim, edgeCount,
                     mMaxNeighbors, mRelaxed, mBeta, mLp, count);
    }

    delete X;
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Actual Pruning: " << std::flush;
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() / 1000. << " s" << std::endl;
}

void Graph::populate_whole()
{
    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "Initial KNN Search " << std::flush;
    mSearchIndex->search(0, mCount, mMaxNeighbors, mEdges, mDistances);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() / 1000. << " s" << std::endl;
    start = std::chrono::high_resolution_clock::now();

    if (mDiscreteSteps > 0)
    {
        nglcu::prune_discrete(mData, mEdges, NULL, mCount, mDim, mCount,
                              mMaxNeighbors, NULL, mDiscreteSteps, mRelaxed, mBeta, mLp);
    }
    else
    {
        nglcu::prune(mData, mEdges, NULL, mCount, mDim, mCount,
                     mMaxNeighbors, mRelaxed, mBeta, mLp);
    }
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Actual Pruning: " << std::flush;
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() / 1000. << " s" << std::endl;
}

void Graph::restart_iteration()
{
    mIterationFinished = false;
    mCurrentCol = 0;
    mCurrentRow = 0;
    // Load up the first edge
    while (mEdges[(mCurrentRow - mRowOffset) * mMaxNeighbors + mCurrentCol] == -1)
    {
        advanceIteration();
    }
}

Edge Graph::next()
{
    Edge e;
    if (mIterationFinished)
    {
        // Set up the next round of iteration
        mIterationFinished = false;
        e.indices[0] = -1;
        e.indices[1] = -1;
        e.distance = 0;
        return e;
    }
    int currentIndex = (mCurrentRow - mRowOffset) * mMaxNeighbors + mCurrentCol;
    e.distance = mDistances[currentIndex];
    if (mReversed)
    {
        e.indices[0] = mEdges[currentIndex];
        e.indices[1] = mCurrentRow;
    }
    else
    {
        e.indices[0] = mCurrentRow;
        e.indices[1] = mEdges[currentIndex];
    }

    mReversed = !mReversed;
    if (!mReversed)
    {
        advanceIteration();
        while (mEdges[(mCurrentRow - mRowOffset) * mMaxNeighbors + mCurrentCol] == -1)
        {
            advanceIteration();
        }
    }

    return e;
}

void Graph::advanceIteration()
{
    mCurrentCol++;
    if (mCurrentCol >= mMaxNeighbors)
    {
        mCurrentCol = 0;
        mCurrentRow++;
        if (mCurrentRow >= mCount)
        {
            mCurrentRow = 0;
            mIterationFinished = true;
        }
        // If we have changed rows, let's ensure we don't need to run
        // another query
        if (mChunked)
        {
            if (mCurrentRow - mRowOffset >= mQuerySize || mCurrentRow == 0)
            {
                populate_chunk(mCurrentRow);
            }
        }
    }
}

Graph::~Graph()
{
    if (mSelfConstructedIndex)
    {
        delete mSearchIndex;
    }
    delete mEdges;
    delete mDistances;
}