#include "kernel.cuh"
#include <cmath>

double host_sphere_func(const double *solution) {
    double sum = 0.0;
    for (int i = 0; i < NUM_OF_DIMENSIONS; ++i) {
        double x = solution[i];
        sum += x * x;
    }
    return sum;
}

double host_rosenbrock_func(const double *solution) {
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

double host_rastrigin_func(const double *solution) {
    double sum = 0.0;
    for (int i = 0; i < NUM_OF_DIMENSIONS; ++i) {
        double sol = (solution[i]);
        sum += (sol * sol - 10.0 * cos(2.0 * phi * sol) + 10.0);
    }
    return sum;
}

double host_ackley_func(const double *solution) {
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

/**
 * Compute the fitness of a solution
 * @param solution Array of positions
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
 * Get a random number between given range
 * @param low Low bound
 * @param high High bound
 * @return random number
 */
double getRandom(double low, double high) {
    return low + ((high - low) + 1) * rand() / (RAND_MAX + 1.0);
}

/**
 * Get a random number between 0.0 and 1.0f
 * @return random number
 */
double getRandomClamped() {
    return static_cast<double>(rand()) / RAND_MAX;
}
