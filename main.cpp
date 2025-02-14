#include <limits>
#include <iomanip>

#include "kernel.h"

int main(const int argc, char **argv) {
    if (argc == 3) {
        int dim = std::stoi(argv[1]);
        int pop = std::stoi(argv[2]);
    }


    double positions[POPULATION_SIZE * NUM_OF_DIMENSIONS];
    double pBests[POPULATION_SIZE * NUM_OF_DIMENSIONS];
    double gBest[NUM_OF_DIMENSIONS];

    printf("Type \t Time \t  \t Minimum\n");

    // Initialisation du random
    srand(static_cast<unsigned>(time(nullptr)));

    for (int i = 0; i < POPULATION_SIZE * NUM_OF_DIMENSIONS; i++) {
        positions[i] = getRandom(START_RANGE_MIN, START_RANGE_MAX);
        pBests[i] = positions[i];
    }

    for (int k = 0; k < NUM_OF_DIMENSIONS; k++)
        gBest[k] = pBests[k];

    const clock_t begin = clock();
    cuda_pso(positions, pBests, gBest); // gBest become the best fitness
    const clock_t end = clock();
    printf("GPU \t ");
    printf("%10.3lf \t", static_cast<double>(end - begin) / CLOCKS_PER_SEC);

    std::cout << std::setprecision(17) << host_fitness_function(gBest);

    return 0;
}
