#include "kernel.cuh"

#include <ctime>
#include <iomanip>
#include <iostream>
#include <string>

int main(const int argc, char **argv) {
    for (int i = 0; i < 10; i++) {
        double positions[POPULATION_SIZE * NUM_OF_DIMENSIONS];
        double pBests[POPULATION_SIZE * NUM_OF_DIMENSIONS];
        double gBest[NUM_OF_DIMENSIONS];

        std::cout << "Type" << " \t "
                << "Time" << " \t "
                << "Minimum"
                << std::endl;

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

        std::cout
                << "GPU" << " \t "
                << std::setprecision(2)
                << static_cast<double>(end - begin) / CLOCKS_PER_SEC << " \t "
                << std::setprecision(4) << host_fitness_function(gBest)
                << std::endl;
    }

    return 0;
}
