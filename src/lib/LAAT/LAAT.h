#ifndef ANTPCA_H
#define ANTPCA_H

#include <vector>
#include <array>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <random>
#include <omp.h>
#include <unordered_map>

#include "Eigen/Core"
#include "Eigen/SVD"
#include "utils/KDTreeVectorOfVectorsAdaptor.h"
#include "neighbourhoodmap.h"

// KDTree for range search
template <size_t Dim>
using KDTree = KDTreeVectorOfVectorsAdaptor<
  std::vector<std::array<float, Dim>>, float>;

/*
  global variable that counts the number of lost ants to possibly give the
  user a warning.
*/
extern size_t lostAnts;
extern size_t G_memory;

// main function for the LAAT algorithm
template <size_t Dim>
std::vector<float> LocallyAlignedAntTechnique(
  std::vector<std::array<float, Dim>> const &data,
  std::vector<size_t> const &ants,
  size_t const numberOfIterations,
  size_t const numberOfSteps,
  size_t const threshold,
  float const memoryLimit,
  float const radius,
  float const beta,
  float const kappa,
  float const p_release,
  float const evapRate,
  float const lowerlimit,
  float const upperlimit,
  bool const showProgress);

// preprocessing functions
template <size_t Dim>
size_t preprocess(std::vector<std::array<float, Dim>> const &data,
		  float const radius,
		  float const threshold,
		  std::vector<unsigned int> &neighbours,
		  std::vector<std::vector<std::array<float, Dim>>> &eigenVectors,
		  std::vector<std::array<float, Dim>> &eigenValues,
		  KDTree<Dim> &kdTree);
template <size_t Dim>
std::vector<unsigned int> groupdata(std::vector<std::array<float, Dim>> const &data,
				    std::array<size_t, Dim> const &ants);
  
// iterative functions
void initializeAnts(std::vector<unsigned int> const &neighbourhoods,
		    std::vector<unsigned int> const &gd,
		    size_t medianValue,
		    std::vector<unsigned int> &antLocations);
template <size_t Dim>
void antSearch(std::vector<std::array<float, Dim>> const &data,
	       std::vector<unsigned int> const &neighbours,
	       NeighbourhoodMap<Dim> const &neighbourhoodMap,	       
	       std::vector<unsigned int> const &antLocations,
	       std::vector<std::vector<std::array<float, Dim>>> const &eigenVectors,
	       std::vector<std::array<float, Dim>> const &eigenValues,
	       KDTree<Dim> &kdTree,
	       size_t const numberOfSteps,
	       float const radius,
	       float const beta,
	       float const kappa,
	       float const p_release,
  	       std::vector<float> &pheromone);
void evaporatePheromone(std::vector<float> &pheromone,
		        float evapRate,
			float lowerlimit,
			float upperlimit);

// functions to communicate with the user
void initializeProgressBar(size_t size);
void updateProgressBar(size_t loop);
void completeProgressBar();
void printWarning(size_t amountRuns);

// auxiliary function to find the median of a vector
size_t median(std::vector<unsigned int> const &data);

// template functions
#include "preprocess.h"
#include "groupdata.h"
#include "antsearch.h"
#include "locallyalignedanttechnique.h"

#endif
