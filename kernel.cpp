#include "kernel.cuh"
#include <cmath>

/**
 * Fonction de coût Sphere exécutée sur le GPU.
 */
double host_sphere_func(const double *solution) {
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
double host_rosenbrock_func(const double *solution) {
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
double host_rastrigin_func(const double *solution) {
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
double host_ackley_func(const double *solution) {
    double sum1 = 0.f;
    double sum2 = 0.f;
    for (int i = 0; i < NUM_OF_DIMENSIONS; ++i) {
        const double sol = (solution[i]);
        sum1 += sol * sol;
        sum2 += cos(2.f * M_PI * sol);
    }
    const double term1 = -20.f * exp(-0.2 * sqrt(sum1 / NUM_OF_DIMENSIONS));
    const double term2 = -exp(sum2 / NUM_OF_DIMENSIONS);
    return term1 + term2 + 20.f + M_E;
}

/**
 * Sélectionne la fonction de coût à utiliser selon la constante SELECTED_OBJ_FUNC.
 * @param solution 0 - Ackley Function, 1 - Rastigrin Function, 2 - Rosenbrock Function, 3 - Sphere Function
 * @return Fitness
 */
double host_fitness_function(const double *solution) {
    switch (SELECTED_OBJ_FUNC) {
        case 0:
            return host_ackley_func(solution);
        case 1:
            return host_rastrigin_func(solution);
        case 2:
            return host_rosenbrock_func(solution);
        default:
            return host_sphere_func(solution);
    }
}

/**
 * Génère un nombre aléatoire dans un intervalle donné.
 * @param low Low bound
 * @param high High bound
 * @return random number
 */
double getRandom(const double low, const double high) {
    return low + ((high - low) + 1) * rand() / (RAND_MAX + 1.f);
}

/**
 * Génère un nombre aléatoire entre 0.0 et 1.0f
 * @return random number
 */
double getRandomClamped() {
    return static_cast<double>(rand()) / RAND_MAX;
}
