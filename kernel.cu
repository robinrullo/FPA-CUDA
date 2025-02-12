#include <cuda_runtime.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <math_constants.h>

#include "kernel.h"

#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

inline void printBests(float *best, int size) {
    printf("Best values: ");
    for (int j = 0; j < size; ++j) {
        printf("%f ", best[j]);
    }
    printf("\n");
}

__device__ float tempSolution1[NUM_OF_DIMENSIONS];
__device__ float tempSolution2[NUM_OF_DIMENSIONS];


__device__ double levy_step(double beta, curandState *state) {
    double sigma_u = pow(
        tgamma(1.0 + beta) * sin(CUDART_PI * beta / 2) /
        (tgamma((1.0 + beta) / 2.0) * beta * pow(2.0, (beta - 1) / 2.0)),
        1.0 / beta);

    constexpr double sigma_v = 1.0;

    // Génération de nombres gaussiens pour u et v
    double u = curand_normal(state) * sigma_u;
    double v = curand_normal(state) * sigma_v;

    // Calcul du pas de Lévy
    return u / pow(fabs(v), 1.0 / beta);
}

/* Objective function
0: Levy 3-dimensional
1: Shifted Rastigrin's Function
2: Shifted Rosenbrock's Function
3: Shifted Griewank's Function
4: Shifted Sphere's Function
*/
/**
 * Runs on the GPU, called from the GPU.
 **/
__device__ float fitness_function(float x[]) {
    float res = 0;
    float somme = 0;
    float produit = 0;

    switch (SELECTED_OBJ_FUNC) {
        case 0: {
            float y1 = 1 + (x[0] - 1) / 4;
            float yn = 1 + (x[NUM_OF_DIMENSIONS - 1] - 1) / 4;

            res += pow(sin(phi * y1), 2);

            for (int i = 0; i < NUM_OF_DIMENSIONS - 1; i++) {
                float y = 1 + (x[i] - 1) / 4;
                float yp = 1 + (x[i + 1] - 1) / 4;
                res += pow(y - 1, 2) * (1 + 10 * pow(sin(phi * yp), 2)) + pow(yn - 1, 2);
            }
            break;
        }
        case 1: {
            for (int i = 0; i < NUM_OF_DIMENSIONS; i++) {
                float zi = x[i] - 0;
                res += pow(zi, 2) - 10 * cos(2 * phi * zi) + 10;
            }
            res -= 330;
            break;
        }
        case 2: {
            for (int i = 0; i < NUM_OF_DIMENSIONS - 1; i++) {
                float zi = x[i] - 0 + 1;
                float zip1 = x[i + 1] - 0 + 1;
                res += 100 * (pow(pow(zi, 2) - zip1, 2)) + pow(zi - 1, 2);
            }
            res += 390;
            break;
        }
        case 3: {
            for (int i = 0; i < NUM_OF_DIMENSIONS; i++) {
                float zi = x[i] - 0;
                somme += pow(zi, 2) / 4000;
                produit *= cos(zi / pow(i + 1, 0.5));
            }
            res = somme - produit + 1 - 180;
            break;
        }
        case 4: {
            for (int i = 0; i < NUM_OF_DIMENSIONS; i++) {
                float zi = x[i] - 0;
                res += pow(zi, 2);
            }
            res -= 450;
            break;
        }
    }

    return res;
}

/***************************************************
 * Runs on the GPU, called from the CPU or the GPU *
 ***************************************************/
/**
 * CUDA kernel to setup the GPU random generator
 **/
__global__ void setupRandomStates(curandState *states, unsigned long seed, int totalThreads) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < totalThreads) {
        curand_init(seed + idx, idx, 0, &states[idx]);
    }
}

__global__ void kernelUpdatePopulation(
    double *positions,
    double *pBests,
    double *gBest,
    int epoch,
    double *steps,
    curandState *states
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    curandState *state = &states[i];

    // avoid an out of bound for the array
    if (i >= POPULATION_SIZE * NUM_OF_DIMENSIONS)
        return;

    double newVal;
    if (curand_uniform(state) < GLOBAL_POLLINATION_RATE) {
        double levy = LEVY_MULTIPLIER * levy_step(LEVY_BETA, state) * curand_normal(state);
        newVal = positions[i] + 1.0 / sqrtf(epoch) * levy * positions[i] - gBest[i % NUM_OF_DIMENSIONS];
        /*printf("[%d] - Levy: %f\n", i, levy);
        printf("[%d] - epoch: %d\n", i, epoch);
        printf("[%d] - position: %f\n", i, positions[i]);
        printf("[%d] - sqrt: %f\n", i, sqrtf(epoch));
        printf("[%d] - gBest: %f\n", i, gBest[i]);
        printf("[%d] - gBest modulo: %f\n", i, gBest[i % NUM_OF_DIMENSIONS]);
        printf("[%d] - newVal: %f\n", i, newVal);*/


    } else {
        uint id1 = curand(state) % POPULATION_SIZE;
        uint id2 = curand(state) % POPULATION_SIZE;
        newVal = positions[i] + curand_uniform(state) * (
                     positions[id1 * NUM_OF_DIMENSIONS + (i % NUM_OF_DIMENSIONS)] - positions[
                         id2 * NUM_OF_DIMENSIONS + (i % NUM_OF_DIMENSIONS)]);
    }
    positions[i] = newVal;
}

/**
 * Runs on the GPU, called from the CPU or the GPU
 **/
/***
 * Update pBest if new value is better than previous epoch else reuse previous
 **/
__global__ void kernelUpdatePBest(double *positions, double *pBests, double *gBest) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

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


extern "C" void cuda_pso(double *positions, double *pBests, double *gBest) {
    int size = POPULATION_SIZE * NUM_OF_DIMENSIONS;

    // declare all the arrays on the device
    double *devPos;
    double *devPBest;
    double *devGBest;
    double *devSteps;

    double temp[NUM_OF_DIMENSIONS];

    curandState *devStates;


    // Memory allocation
    cudaMalloc(&devPos, size * sizeof(float));
    cudaMalloc(&devPBest, size * sizeof(float));
    cudaMalloc(&devGBest, NUM_OF_DIMENSIONS * sizeof(float));
    cudaMalloc(&devStates, size * sizeof(curandState));
    cudaMalloc(&devSteps, size * sizeof(double));


    // Thread & Block number
    int threadsNum = 32;
    int blocksNum = ceil(size / threadsNum);

    // Copy particle datas from host to device
    /**
     * Copy in GPU memory the data from the host
     **/
    cudaMemcpy(devPos, positions, sizeof(float) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(devPBest, pBests, sizeof(float) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(devGBest, gBest, sizeof(float) * NUM_OF_DIMENSIONS, cudaMemcpyHostToDevice);

    setupRandomStates<<<blocksNum, threadsNum>>>(devStates, time(NULL), size);
    gpuErrorCheck(cudaPeekAtLastError());
    gpuErrorCheck(cudaDeviceSynchronize());

    // PSO main function
    // MAX_EPOCHS = 30000;

    for (int iter = 0; iter < MAX_EPOCHS; iter++) {
        kernelUpdatePopulation<<<blocksNum, threadsNum>>>(
            devPos,
            devPBest, devGBest,
            iter + 1,
            devSteps,
            devStates
        );

        kernelUpdatePBest<<<blocksNum, threadsNum>>>(devPos, devPBest, devGBest);

        cudaMemcpy(pBests, devPBest, sizeof(float) * POPULATION_SIZE * NUM_OF_DIMENSIONS, cudaMemcpyDeviceToHost);

        //printBests(pBests, POPULATION_SIZE);

        for (int i = 0; i < size; i += NUM_OF_DIMENSIONS) {
            for (int k = 0; k < NUM_OF_DIMENSIONS; k++) //ssB1
                temp[k] = pBests[i + k];

            if (host_fitness_function(temp) < host_fitness_function(gBest)) {
                for (int k = 0; k < NUM_OF_DIMENSIONS; k++)
                    gBest[k] = temp[k];
            }
        }

        cudaMemcpy(devGBest, gBest, sizeof(float) * NUM_OF_DIMENSIONS, cudaMemcpyHostToDevice);
    }

    cudaMemcpy(positions, devPos, sizeof(float) * size, cudaMemcpyDeviceToHost);
    cudaMemcpy(pBests, devPBest, sizeof(float) * size, cudaMemcpyDeviceToHost);
    cudaMemcpy(gBest, devGBest, sizeof(float) * NUM_OF_DIMENSIONS, cudaMemcpyDeviceToHost);


    // cleanup
    cudaFree(devPos);
    cudaFree(devPBest);
    cudaFree(devGBest);
    cudaFree(devStates);
    cudaFree(devSteps);
}
