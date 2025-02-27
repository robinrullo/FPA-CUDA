#include "kernel.cuh"

#include <ctime>
#include <iomanip>
#include <iostream>
#include <string>

/**
 * Programme principal qui exécute l'algorithme d'optimisation FPA sur GPU.
 * Il génère une population initiale, exécute l'optimisation, et affiche les résultats.
 */
int main(const int argc, char **argv) {
    for (int i = 0; i < 10; i++) { // Réalisation de 10 itérations
        // Déclaration des tableaux pour les positions des pollens
        double positions[POPULATION_SIZE * NUM_OF_DIMENSIONS];
        double pBests[POPULATION_SIZE * NUM_OF_DIMENSIONS];
        double gBest[NUM_OF_DIMENSIONS];

        // Affichage des en-têtes des résultats
        std::cout << "Type" << " \t "
                << "Time" << " \t "
                << "Minimum"
                << std::endl;

        // Initialisation du générateur de nombres aléatoires
        srand(static_cast<unsigned>(time(nullptr)));

        // Initialisation des positions des pollens
        for (int j = 0; j < POPULATION_SIZE * NUM_OF_DIMENSIONS; j++) {
            positions[j] = getRandom(START_RANGE_MIN, START_RANGE_MAX);
            pBests[j] = positions[j];
        }

        // Initialisation du meilleur individu global avec le premier pollen
        for (int k = 0; k < NUM_OF_DIMENSIONS; k++)
            gBest[k] = pBests[k];

        // Mesure du temps d'exécution de l'algorithme sur GPU
        const clock_t begin = clock();
        cuda_fpa(positions, pBests, gBest);
        const clock_t end = clock();

        // Affichage des résultats
        std::cout
                << "GPU" << " \t "
                << std::setprecision(2)
                << static_cast<double>(end - begin) / CLOCKS_PER_SEC << " \t "
                << std::setprecision(4) << host_fitness_function(gBest)
                << std::endl;
    }

    return 0;
}
