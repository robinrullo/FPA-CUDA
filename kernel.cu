#include <iostream>

#include <cuda_runtime.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <math_constants.h>

#include "kernel.cuh"

#include <iomanip>

#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(const cudaError_t code, const char *file, const int line, const bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__device__ double tempSolution1[NUM_OF_DIMENSIONS];
__device__ double tempSolution2[NUM_OF_DIMENSIONS];


/**
 * Runs on the GPU, called from the GPU.
 **/
__device__ double sphere_func(const double *solution) {
    double sum = 0.0;
    for (int i = 0; i < NUM_OF_DIMENSIONS; ++i) {
        double x = solution[i];
        sum += x * x;
    }
    //printf("sum: %f\n", sum);
    return sum;
}

__device__ double rosenbrock_func(const double *solution) {
    double sum = 0.0;
    for (int i = 0; i < NUM_OF_DIMENSIONS - 1; ++i) {
        double xi = (solution[i]);
        double xi1 = (solution[i + 1]);
        double term1 = 100.0 * pow(xi1 - pow(xi, 2.0), 2.0);
        double term2 = pow(1.0 - xi, 2.0);
        sum += term1 + term2;
    }
    return sum;
}

__device__ double rastrigin_func(const double *solution) {
    double sum = 0.0;
    for (int i = 0; i < NUM_OF_DIMENSIONS; ++i) {
        double sol = (solution[i]);
        sum += (sol * sol - 10.0 * cos(2.0 * phi * sol) + 10.0);
    }
    return sum;
}

__device__ double ackley_func(const double *solution) {
    double sum1 = 0.0;
    double sum2 = 0.0;
    for (int i = 0; i < NUM_OF_DIMENSIONS; ++i) {
        double sol = (solution[i]);
        sum1 += sol * sol;
        sum2 += cos(2.0 * phi * sol);
    }
    double term1 = -20.0 * exp(-0.2 * sqrt(sum1 / NUM_OF_DIMENSIONS));
    double term2 = -exp(sum2 / NUM_OF_DIMENSIONS);
    return term1 + term2 + 20.0 + M_E;
}

/* Objective function
0: Levy 3-dimensional
1: Shifted Rastigrin's Function
2: Shifted Rosenbrock's Function
3: Shifted Griewank's Function
4: Shifted Sphere's Function
*/
__device__ double fitness_function(const double *solution) {
    switch (SELECTED_OBJ_FUNC) {
        case 0:
            return ackley_func(solution);
        case 1:
            return rastrigin_func(solution);
        case 2:
            return rosenbrock_func(solution);
        default:
            return sphere_func(solution);
    }
}

/***************************************************
 * Runs on the GPU, called from the CPU or the GPU *
 ***************************************************/
/**
 * CUDA kernel to setup the GPU random generator
 **/
__global__ void setupRandomStates(curandState *states, const unsigned long seed, const int totalThreads) {
    const uint idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < totalThreads) {
        curand_init(seed + idx, idx, 0, &states[idx]);
    }
}

__global__ void kernelEnsureNewPosBounds(double *positions, curandState *states) {
    const uint i = blockIdx.x * blockDim.x + threadIdx.x;
    curandState *state = &states[i];
    if (i >= POPULATION_SIZE * NUM_OF_DIMENSIONS) return;

    if (positions[i] < START_RANGE_MIN || positions[i] > START_RANGE_MAX) {
        positions[i] = START_RANGE_MIN;
    }
}

__global__ void kernelUpdatePopulation(
    double *positions,
    const double *pBests,
    const double *gBest,
    const int epoch,
    curandState *states
) {
    const uint i = blockIdx.x * blockDim.x + threadIdx.x;
    curandState *state = &states[i];

    // avoid an out of bound for the array
    if (i >= POPULATION_SIZE * NUM_OF_DIMENSIONS)
        return;

    if (curand_uniform(state) < GLOBAL_POLLINATION_RATE) {
        const double sigma = pow((tgamma(1 + LEVY_BETA) * sin(CUDART_PI * LEVY_BETA / 2)) /
                                 (tgamma((1 + LEVY_BETA) / 2) * LEVY_BETA * pow(2, (LEVY_BETA - 1) / 2)),
                                 (1 / LEVY_BETA));

        // Génération des valeurs u et v
        const double u = curand_normal_double(state) * sigma; // normal distribution
        const double v = curand_uniform_double(state) + 1e-10; // uniform distribution and avoid div by 0

        // Compute levy steps
        const double levy = LEVY_MULTIPLIER * (u / pow(fabs(v), (1 / LEVY_BETA)));

        // Mise à jour de la position
        //positions[i] = positions[i] + 1. / sqrtf(epoch) * levy * (pBests[i] - gBest[i % NUM_OF_DIMENSIONS]);
        //positions[i] = positions[i] + levy * (pBests[i] - gBest[i % NUM_OF_DIMENSIONS]);
        positions[i] = pBests[i] + levy * (pBests[i] - gBest[i % NUM_OF_DIMENSIONS]);

        /*printf("[%d-%d-LEVY] - Levy: %f | epoch: %d | position: %f | sqrt: %f | gBest: %f\n",
               epoch, i, levy, epoch, positions[i], sqrtf(epoch), gBest[i % NUM_OF_DIMENSIONS]);*/
    } else {
        const uint id1 = curand(state) % POPULATION_SIZE;
        const uint id2 = curand(state) % POPULATION_SIZE;

        positions[i] = pBests[i] + curand_uniform_double(state) * (
                           pBests[id1 * NUM_OF_DIMENSIONS + (i % NUM_OF_DIMENSIONS)] -
                           pBests[id2 * NUM_OF_DIMENSIONS + (i % NUM_OF_DIMENSIONS)]
                       );

        /*printf(
            "[%d-%d-SELF] - id1=%d | id2=%d | posId1=%d | posId2=%d | pos1=%f | pos2=%f | newPos=%f | curand_uniform=%f\n",
            epoch,
            i,
            id1,
            id2,
            id1 * NUM_OF_DIMENSIONS + (i % NUM_OF_DIMENSIONS),
            id2 * NUM_OF_DIMENSIONS + (i % NUM_OF_DIMENSIONS),
            positions[id1 * NUM_OF_DIMENSIONS + (i % NUM_OF_DIMENSIONS)],
            positions[id2 * NUM_OF_DIMENSIONS + (i % NUM_OF_DIMENSIONS)],
            newPos,
            curand_uniform(state)
        );
        positions[i] = newPos;*/
    }
}

/**
 * Runs on the GPU, called from the CPU or the GPU
 **/
/***
 * Update pBest if new value is better than previous epoch else reuse previous
 **/
__global__ void kernelUpdatePBest(const double *positions, double *pBests, double *gBest) {
    const uint i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= POPULATION_SIZE * NUM_OF_DIMENSIONS || i % NUM_OF_DIMENSIONS != 0)
        return;

    for (int j = 0; j < NUM_OF_DIMENSIONS; j++) {
        tempSolution1[j] = positions[i + j];
        tempSolution2[j] = pBests[i + j];
    }

    if (fitness_function(tempSolution1) < fitness_function(tempSolution2)) {
        for (int k = 0; k < NUM_OF_DIMENSIONS; k++)
            pBests[i + k] = positions[i + k];
    }
}

__global__ void kernelUpdateGBest(const double *pBests, double *gBest) {
    const uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= POPULATION_SIZE * NUM_OF_DIMENSIONS || i % NUM_OF_DIMENSIONS != 0) return;

    double temp[NUM_OF_DIMENSIONS];

    for (int k = 0; k < NUM_OF_DIMENSIONS; k++)
        temp[k] = pBests[i + k];

    //printf("[%d] - Fitness function: %f\n", i, fitness_function(temp));
    double gBestFitness = fitness_function(gBest);
    double tempBestFitness = fitness_function(temp);

    if (tempBestFitness < gBestFitness) {
        for (int k = 0; k < NUM_OF_DIMENSIONS; k++)
            gBest[k] = temp[k];
    }
}

void cuda_pso(double *positions, double *pBests, double *gBest) {
    constexpr int size = POPULATION_SIZE * NUM_OF_DIMENSIONS;

    // declare all the arrays on the device
    double *devPos;
    double *devPBest;
    double *devGBest;

    double temp[NUM_OF_DIMENSIONS];

    curandState *devStates;


    // Memory allocation
    cudaMalloc(&devPos, size * sizeof(double));
    cudaMalloc(&devPBest, size * sizeof(double));
    cudaMalloc(&devGBest, NUM_OF_DIMENSIONS * sizeof(double));
    cudaMalloc(&devStates, size * sizeof(curandState));


    // Thread & Block number
    int threadsNum = 32;
    int blocksNum = ceil(size / (double) threadsNum);

    // Copy particle datas from host to device
    /**
     * Copy in GPU memory the data from the host
     **/
    cudaMemcpy(devPos, positions, sizeof(double) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(devPBest, pBests, sizeof(double) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(devGBest, gBest, sizeof(double) * NUM_OF_DIMENSIONS, cudaMemcpyHostToDevice);

    setupRandomStates<<<blocksNum, threadsNum>>>(devStates, time(nullptr), size);
    gpuErrorCheck(cudaPeekAtLastError());
    gpuErrorCheck(cudaDeviceSynchronize());

    for (int iter = 0; iter < MAX_EPOCHS; iter++) {
        kernelUpdatePopulation<<<blocksNum, threadsNum>>>(
            devPos,
            devPBest,
            devGBest,
            iter + 1,
            devStates
        );
        gpuErrorCheck(cudaPeekAtLastError());
        gpuErrorCheck(cudaDeviceSynchronize());

        kernelEnsureNewPosBounds<<<blocksNum, threadsNum>>>(devPos, devStates);
        gpuErrorCheck(cudaPeekAtLastError());
        gpuErrorCheck(cudaDeviceSynchronize());

        kernelUpdatePBest<<<blocksNum, threadsNum>>>(devPos, devPBest, devGBest);
        gpuErrorCheck(cudaPeekAtLastError());
        gpuErrorCheck(cudaDeviceSynchronize());

        kernelUpdateGBest<<<blocksNum, threadsNum>>>(devPBest, devGBest);
        gpuErrorCheck(cudaPeekAtLastError());
        gpuErrorCheck(cudaDeviceSynchronize());

        //cudaMemcpy(pBests, devPBest, sizeof(double) * POPULATION_SIZE * NUM_OF_DIMENSIONS, cudaMemcpyDeviceToHost);
        cudaMemcpy(gBest, devGBest, sizeof(double) * NUM_OF_DIMENSIONS, cudaMemcpyDeviceToHost);
        gpuErrorCheck(cudaPeekAtLastError());
        gpuErrorCheck(cudaDeviceSynchronize());
        for (int i = 0; i < size; i += NUM_OF_DIMENSIONS) {
            for (int k = 0; k < NUM_OF_DIMENSIONS; k++)
                temp[k] = pBests[i + k];

            //printf("[%d] - Fitness function: %f\n", i, host_fitness_function(temp));
            if (host_fitness_function(temp) < host_fitness_function(gBest)) {
                for (int k = 0; k < NUM_OF_DIMENSIONS; k++)
                    gBest[k] = temp[k];
            }
        }

        //cudaMemcpy(devGBest, gBest, sizeof(double) * NUM_OF_DIMENSIONS, cudaMemcpyHostToDevice);
        cudaMemcpy(devPBest, pBests, sizeof(double) * NUM_OF_DIMENSIONS, cudaMemcpyHostToDevice);
        gpuErrorCheck(cudaPeekAtLastError());
        gpuErrorCheck(cudaDeviceSynchronize());
    }

    cudaMemcpy(positions, devPos, sizeof(double) * size, cudaMemcpyDeviceToHost);
    cudaMemcpy(pBests, devPBest, sizeof(double) * size, cudaMemcpyDeviceToHost);
    cudaMemcpy(gBest, devGBest, sizeof(double) * NUM_OF_DIMENSIONS, cudaMemcpyDeviceToHost);


    // cleanup
    cudaFree(devPos);
    cudaFree(devPBest);
    cudaFree(devGBest);
    cudaFree(devStates);
}
