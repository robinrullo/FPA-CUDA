#include "kernel.h"

/* Objective function
0: Levy 3-dimensional
1: Shifted Rastigrin's Function
2: Shifted Rosenbrock's Function
3: Shifted Griewank's Function
4: Shifted Sphere's Function
*/
// parametre 1 individu avec ses positions
/**
 * Runs on the GPU, called from the GPU.
 **/
double host_sphere_func(const double *solution) {
    double sum = 0.0;
    for (int i = 0; i < NUM_OF_DIMENSIONS; ++i) {
        double x = solution[i];
        sum += x * x;
    }
    //printf("sum: %f\n", sum);
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

/* Objective function
0: Levy 3-dimensional
1: Shifted Rastigrin's Function
2: Shifted Rosenbrock's Function
3: Shifted Griewank's Function
4: Shifted Sphere's Function
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

// Obtenir un random entre low et high
double getRandom(double low, double high) {
    return low + ((high - low) + 1) * rand() / (RAND_MAX + 1.0);
}

// Obtenir un random entre 0.0f and 1.0f inclusif
double getRandomClamped() {
    return static_cast<double>(rand()) / RAND_MAX;
}
