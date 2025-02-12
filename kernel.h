#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <iostream>
#include <string>


// Constantes
/* Objective function
0: Levy 3-dimensional
1: Shifted Rastigrin's Function
2: Shifted Rosenbrock's Function
3: Shifted Griewank's Function
4: Shifted Sphere's Function
*/
const int SELECTED_OBJ_FUNC = 0;

/*const int POPULATION_SIZE = 30;
const int NUM_OF_DIMENSIONS = 3;
const int MAX_EPOCHS = 1;*/

const int POPULATION_SIZE = 30; // 30, 50, 70
const int NUM_OF_DIMENSIONS = 10; // 10, 30, 50
const int MAX_EPOCHS = 5 * pow(10, 3); //NUM_OF_DIMENSIONS * pow(10, 4);
constexpr float GLOBAL_POLLINATION_RATE = .8f; // Global pollination probability
constexpr float LEVY_BETA = 1.0;
constexpr float LEVY_MULTIPLIER = 0.2;
const float START_RANGE_MIN = -5.12f;
const float START_RANGE_MAX = 5.12f;
const float phi = 3.1415;

// Les 3 fonctions très utiles
double getRandom(double low, double high);

double getRandomClamped();

double host_fitness_function(const double x[]);

// Fonction externe qui va tourner sur le GPU
extern "C" void cuda_pso(double *positions, double *pBests, double *gBest);
