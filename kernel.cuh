#ifndef MAIN_CUH
#define MAIN_CUH

#include "cmath"

// Constantes
/* Objective function
0: host_ackley_func(solution);
1: host_rastrigin_func(solution);
2: host_rosenbrock_func(solution);
default: host_sphere_func(solution);
*/
const int SELECTED_OBJ_FUNC = 1;

const int POPULATION_SIZE = 30; // 30, 50, 70
const int NUM_OF_DIMENSIONS = 10; // 10, 30, 50
const int MAX_EPOCHS = 5 * pow(10, 3);
constexpr float GLOBAL_POLLINATION_RATE = .8f;
constexpr float LEVY_BETA = 1.0;
constexpr float LEVY_MULTIPLIER = 0.2;
const float START_RANGE_MIN = -5.12f;
const float START_RANGE_MAX = 5.12f;
const float phi = 3.1415;


double getRandom(double low, double high);

double getRandomClamped();

double host_fitness_function(const double x[]);

// Fonction externe qui va tourner sur le GPU
extern "C" void cuda_pso(double *positions, double *pBests, double *gBest);
#endif // MAIN_CUH
