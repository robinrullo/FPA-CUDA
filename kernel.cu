#include <iostream>

#include <cuda_runtime.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <math_constants.h>

#include "kernel.cuh"

#include <iomanip>

#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }

/**
 * Vérifie les erreurs CUDA et affiche un message si une erreur est détectée.
 */
inline void gpuAssert(const cudaError_t code, const char *file, const int line, const bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Variables globales pour stocker temporairement les solutions sur le GPU
__device__ double tempSolution1[NUM_OF_DIMENSIONS];
__device__ double tempSolution2[NUM_OF_DIMENSIONS];


/**
 * Fonction de coût Sphere exécutée sur le GPU.
 */
__device__ double sphere_func(const double *solution) {
    double sum = 0.0;
    for (int i = 0; i < NUM_OF_DIMENSIONS; ++i) {
        const double x = solution[i];
        sum += x * x;
    }
    return sum;
}

/**
 * Fonction de coût Rosenbrock exécutée sur le GPU.
 */
__device__ double rosenbrock_func(const double *solution) {
    double sum = 0.0;
    for (int i = 0; i < NUM_OF_DIMENSIONS - 1; ++i) {
        const double xi = (solution[i]);
        const double xi1 = (solution[i + 1]);
        const double term1 = 100.0 * pow(xi1 - pow(xi, 2.0), 2.0);
        const double term2 = pow(1.0 - xi, 2.0);
        sum += term1 + term2;
    }
    return sum;
}

/**
 * Fonction de coût Rastrigin exécutée sur le GPU.
 */
__device__ double rastrigin_func(const double *solution) {
    double sum = 0.0;
    for (int i = 0; i < NUM_OF_DIMENSIONS; ++i) {
        const double sol = (solution[i]);
        sum += (sol * sol - 10.0 * cos(2.0 * M_PI * sol) + 10.0);
    }
    return sum;
}

/**
 * Fonction de coût Ackley exécutée sur le GPU.
 */
__device__ double ackley_func(const double *solution) {
    double sum1 = 0.0;
    double sum2 = 0.0;
    for (int i = 0; i < NUM_OF_DIMENSIONS; ++i) {
        const double sol = (solution[i]);
        sum1 += sol * sol;
        sum2 += cos(2.0 * M_PI * sol);
    }
    const double term1 = -20.0 * exp(-0.2 * sqrt(sum1 / NUM_OF_DIMENSIONS));
    const double term2 = -exp(sum2 / NUM_OF_DIMENSIONS);
    return term1 + term2 + 20.0 + M_E;
}

/**
 * Sélectionne la fonction de coût à utiliser selon la constante SELECTED_OBJ_FUNC.
 * @param solution 0 - Ackley Function, 1 - Rastigrin Function, 2 - Rosenbrock Function, 3 - Sphere Function
 * @return Fitness
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
 * Initialise les générateurs aléatoires CUDA pour chaque thread.
 */
__global__ void setupRandomStates(curandState *states, const unsigned long seed, const int totalThreads) {
    if (const uint idx = threadIdx.x + blockIdx.x * blockDim.x; idx < totalThreads) {
        curand_init(seed + idx, idx, 0, &states[idx]);
    }
}

/**
 * Met à jour la position des particules en appliquant l'algorithme PSO avec Lévy flight.
 */
__global__ void kernelEnsureNewPosBounds(double *positions) {
    const uint i = blockIdx.x * blockDim.x + threadIdx.x;
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

    if (i >= POPULATION_SIZE * NUM_OF_DIMENSIONS)
        return;

    curandState *state = &states[i];

    // Pollinisation Croisée - Mise à jour selon la règle du Lévy flight
    if (curand_uniform(state) < GLOBAL_POLLINATION_RATE) {
        const double sigma = pow((tgamma(1 + LEVY_BETA) * sin(CUDART_PI * LEVY_BETA / 2)) /
                                 (tgamma((1 + LEVY_BETA) / 2) * LEVY_BETA * pow(2, (LEVY_BETA - 1) / 2)),
                                 (1 / LEVY_BETA));

        // Génération des valeurs u et v
        const double u = curand_normal_double(state) * sigma;
        const double v = curand_uniform_double(state) + 1e-10;

        // Calcul du pas de Levy
        const double levy = LEVY_MULTIPLIER * (u / pow(fabs(v), (1 / LEVY_BETA)));

        // Mise à jour de la position avec le vol de Lévi
        positions[i] = pBests[i] + levy * (pBests[i] - gBest[i % NUM_OF_DIMENSIONS]);
    } else {
        // Autopollinisation
        const uint id1 = curand(state) % POPULATION_SIZE; // Index de la première fleur choisie au hasard
        const uint id2 = curand(state) % POPULATION_SIZE; // Index de la seconde fleur choisie au hasard

        positions[i] = pBests[i] + curand_uniform_double(state) * (
                           pBests[id1 * NUM_OF_DIMENSIONS + (i % NUM_OF_DIMENSIONS)] -
                           pBests[id2 * NUM_OF_DIMENSIONS + (i % NUM_OF_DIMENSIONS)]
                       );
    }
}

/**
 * Runs on the GPU, called from the CPU or the GPU
 **/
/***
 * Update pBest if new value is better than previous epoch else reuse previous
 **/
__global__ void kernelUpdatePBest(const double *positions, double *pBests) {
    const uint idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Uniquement à la taille de la population (sans la dimension)
    if (idx >= POPULATION_SIZE * NUM_OF_DIMENSIONS || idx % NUM_OF_DIMENSIONS != 0)
        return;

    // Copie les positions de la solution actuelle et de la meilleure solution connue
    for (int j = 0; j < NUM_OF_DIMENSIONS; j++) {
        tempSolution1[j] = positions[idx + j];
        tempSolution2[j] = pBests[idx + j];
    }

    // Mise à jour de pBest si la nouvelle position est meilleure
    if (fitness_function(tempSolution1) < fitness_function(tempSolution2)) {
        for (int k = 0; k < NUM_OF_DIMENSIONS; k++)
            pBests[idx + k] = positions[idx + k];
    }
}

/**
 * Met à jour la meilleure solution globale gBest en parcourant pBests.
 * Seule la première dimension de chaque particule est traitée pour éviter des mises à jour redondantes.
 */
__global__ void kernelUpdateGBest(const double *pBests, double *gBest) {
    const uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= POPULATION_SIZE * NUM_OF_DIMENSIONS || i % NUM_OF_DIMENSIONS != 0) return;

    double temp[NUM_OF_DIMENSIONS];

    // Copier la solution actuelle depuis pBests
    for (int k = 0; k < NUM_OF_DIMENSIONS; k++)
        temp[k] = pBests[i + k];

    // Mise à jour de gBest si une meilleure solution est trouvée
    if (fitness_function(temp) < fitness_function(gBest)) {
        for (int k = 0; k < NUM_OF_DIMENSIONS; k++)
            gBest[k] = temp[k];
    }
}

/**
 * Fonction Principale
 * Implémente l'algorithme Flower Pollination Algorithm (FPA) sur GPU.
 * Gère la mémoire et exécute les noyaux CUDA nécessaires pour l'optimisation.
 */
extern "C" void cuda_fpa(double *positions, double *pBests, double *gBest) {
    constexpr int size = POPULATION_SIZE * NUM_OF_DIMENSIONS;

    // Déclaration des pointeurs de mémoire sur le GPU
    double *devPos;
    double *devPBest;
    double *devGBest;

    double temp[NUM_OF_DIMENSIONS];

    curandState *devStates;


    // Allocation mémoire sur le GPU
    cudaMalloc(&devPos, size * sizeof(double));
    cudaMalloc(&devPBest, size * sizeof(double));
    cudaMalloc(&devGBest, NUM_OF_DIMENSIONS * sizeof(double));
    cudaMalloc(&devStates, size * sizeof(curandState));


    // Définition des dimensions des blocs et grilles CUDA
    int threadsNum = 32;
    int blocksNum = ceil(size / static_cast<double>(threadsNum));

    // Copie des données depuis l'hôte vers le GPU
    cudaMemcpy(devPos, positions, sizeof(double) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(devPBest, pBests, sizeof(double) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(devGBest, gBest, sizeof(double) * NUM_OF_DIMENSIONS, cudaMemcpyHostToDevice);

    // Initialisation des générateurs aléatoire sur GPU
    setupRandomStates<<<blocksNum, threadsNum>>>(devStates, time(nullptr), size);
    gpuErrorCheck(cudaPeekAtLastError());
    gpuErrorCheck(cudaDeviceSynchronize());

    // Boucle principale de l'algorithme
    for (int iter = 0; iter < MAX_EPOCHS; iter++) {
        // Mise à jour des positions des pollens
        kernelUpdatePopulation<<<blocksNum, threadsNum>>>(
            devPos,
            devPBest,
            devGBest,
            iter + 1,
            devStates
        );
        gpuErrorCheck(cudaPeekAtLastError());
        gpuErrorCheck(cudaDeviceSynchronize());

        // Assurer que les nouvelles positions sont dans les bornes autorisées
        kernelEnsureNewPosBounds<<<blocksNum, threadsNum>>>(devPos);
        gpuErrorCheck(cudaPeekAtLastError());
        gpuErrorCheck(cudaDeviceSynchronize());

        // Mise à jour du meilleur pBest pour chaque particule
        kernelUpdatePBest<<<blocksNum, threadsNum>>>(devPos, devPBest);
        gpuErrorCheck(cudaPeekAtLastError());
        gpuErrorCheck(cudaDeviceSynchronize());

        // Mise à jour du meilleur individu global
        kernelUpdateGBest<<<blocksNum, threadsNum>>>(devPBest, devGBest);
        gpuErrorCheck(cudaPeekAtLastError());
        gpuErrorCheck(cudaDeviceSynchronize());

        // Copier gBest mis à jour du GPU vers l'hôte
        cudaMemcpy(gBest, devGBest, sizeof(double) * NUM_OF_DIMENSIONS, cudaMemcpyDeviceToHost);
        for (int i = 0; i < size; i += NUM_OF_DIMENSIONS) {
            for (int k = 0; k < NUM_OF_DIMENSIONS; k++)
                temp[k] = pBests[i + k];

            // Vérifier si une meilleure solution existe et mettre à jour gBest
            if (host_fitness_function(temp) < host_fitness_function(gBest)) {
                for (int k = 0; k < NUM_OF_DIMENSIONS; k++)
                    gBest[k] = temp[k];
            }
        }

        // Copier pBest mis à jour de l'hôte vers le GPU
        cudaMemcpy(devPBest, pBests, sizeof(double) * NUM_OF_DIMENSIONS, cudaMemcpyHostToDevice);
    }

    // Copie des résultats finaux vers l'hôte
    cudaMemcpy(positions, devPos, sizeof(double) * size, cudaMemcpyDeviceToHost);
    cudaMemcpy(pBests, devPBest, sizeof(double) * size, cudaMemcpyDeviceToHost);
    cudaMemcpy(gBest, devGBest, sizeof(double) * NUM_OF_DIMENSIONS, cudaMemcpyDeviceToHost);


    // Libération de la mémoire GPU
    cudaFree(devPos);
    cudaFree(devPBest);
    cudaFree(devGBest);
    cudaFree(devStates);
}
