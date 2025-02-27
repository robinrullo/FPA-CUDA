#ifndef MAIN_CUH
#define MAIN_CUH

#include "cmath"

/**
 * Objective function Selection
 *
 * 0 - Ackley Function
 * 1 - Rastigrin Function
 * 2 - Rosenbrock Function
 * 3(default) - Sphere Function
 */
constexpr int SELECTED_OBJ_FUNC = 3;
// Population Size
constexpr int POPULATION_SIZE = 30; // 30, 50, 70
// Number of dimensions
constexpr int NUM_OF_DIMENSIONS = 10; // 10, 30, 50
// Max epochs
constexpr int MAX_EPOCHS = 5e3;
// Global Pollination Rate
constexpr float GLOBAL_POLLINATION_RATE = .8f;
// Levy BETA
constexpr float LEVY_BETA = 1.f;
// Levy Multiplier
constexpr float LEVY_MULTIPLIER = .2f;
// Solutions Lower bound
constexpr float START_RANGE_MIN = -5.12f;
// Solutions Upper bound
constexpr float START_RANGE_MAX = 5.12f;

/**
 * Get a random number between given range
 * @param low Low bound
 * @param high High bound
 * @return random number
 */
double getRandom(double low, double high);

/**
 * Gén 0.0 and 1.0f
 * @return random number
 */
double getRandomClamped();

double host_fitness_function(const double *solution);

/**
 *
 * @param positions Les positions du pollen
 * @param pBests Meilleurs individus locaux
 * @param gBest Meilleur individu global
 */
extern "C" void cuda_fpa(double *positions, double *pBests, double *gBest);
#endif // MAIN_CUH
