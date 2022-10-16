#ifndef LOCALLYALIGNEDANTTECHNIQUE_H
#define LOCALLYALIGNEDANTTECHNIQUE_H

#include <iostream>

/**
 * Implementation of the Ant Colony algorithm proposed by Taghribi et al.
 * (2020).
 * See Algorithm 1 in the paper for a pseudocode of this algorithm.
 *
 * In this function and all of the functions evoked by this one we will refer
 * to the formulas as described by Taghribi et al. (2020).
 *
 * Reference:
 * Taghribi, A., Bunte, K., Smith, R., Shin, J., Mastropietro, M., Peletier,
 * R. F., & Tino, P. (2020). LAAT: Locally Aligned Ant Technique for detecting
 * manifolds of varying density. arXiv preprint arXiv:2009.08326.
 *
 * @param Dim template parameter indicating the dimensionality of the data
 * @param data vector containing all the data points to search, each data point
 *   must be an std::array with a size conforming to the dimensionality of the
 *   data
 * @param ants an std::array conforming to the dimensionality of the data that
 *   indicates how many ants should be distributed along each dimension. The
 *   total number of ants will be equal to the product of the elements of this
 *   array
 * @param numberOfIterations number of tiems to apply ant search
 * @param numberOfSteps number of steps to run each ant at each iteration
 * @param threshold minimum number of nearest neighbours needed for a data point
 *   to be considered
 * @param memoryLimit amount of memory in GB to use for memoization of local
 *   neighbourhoods
 * @param radius radius within which to search for nearest neighbours
 * @param beta the inverse temperature as used in formula (8)
 * @param kappa tuning parameter for the relative importance of the influence
 * of the alignment and pheromone terms as used in formula (7)
 * @param p_release amount of pheromone to release on a data point at each 
 *   visit
 * @param evapRate rate at which to evaporate pheromone after each application
 *   of ant search
 * @param lowerlimit lower limit on the amount of pheromone of a data point
 * @param upperlimit upper limit on the amount of pheromone of a data point
 * @param showProgess whether or not to show the progress of the algorithm
 * @return vector containing the amount of pheromone for each data point after
 *   the application of LAAT
 */
template <size_t Dim>
std::vector<float> LocallyAlignedAntTechnique(
  std::vector<std::array<float, Dim>> const &data,
  std::array<size_t, Dim> const &ants,
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
  bool const showProgess)
{
  if (!showProgess)
    // suppress output
    std::cout.setstate(std::ios_base::failbit);
  
  std::cout << "Running the Locally Aligned Ant Technique algorithm on "
	    << data.size() << " data points\n\n";

  if (threshold < Dim)
  {
    std::cout.clear();
    std::cout << "Error (LAAT): threshold hyper-parameter should be no smaller than "
	      << "the dimensionality of the data\nAborting program...\n\n";
    exit(1);
  }

  // set amount of lost ants to zero, only used to provide warnings
  lostAnts = 0;

  // initialize parallelization with a default of 8 threads
  omp_set_num_threads(8);
  Eigen::initParallel();

  std::vector<unsigned int> neighbours;
  std::vector<std::vector<std::array<float, Dim>>> eigenVectors;
  std::vector<std::array<float, Dim>> eigenValues;
  KDTree<Dim> kdTree(Dim, data);
  kdTree.index->buildIndex();
  
  // preprocess
  std::cout << "Preprocessing...\n";
  size_t medianvalue = preprocess(data, radius, threshold, neighbours,
				  eigenVectors, eigenValues, kdTree);

  // group the data into a corresponding sector for each ant
  std::vector<unsigned int> gd =
    groupdata(data, ants);

  // iterative step
  std::cout << "\nPerforming ant search...\n";
  
  // initial locations of all ants at each iteration
  size_t numAnts = std::accumulate(
    ants.begin(), ants.end(), 1, std::multiplies<size_t>());
  std::vector<unsigned int> antLocations(numAnts);

  // initialize pheromone value for each data point to 1
  std::vector<float> pheromone(data.size(), 1.0);

  // map to store neighbourhoods of data points with largest neighbourhoods
  NeighbourhoodMap neighbourhoodMap(memoryLimit,
				    data,
				    neighbours,
				    kdTree,
				    radius);

  initializeProgressBar(numberOfIterations / 2);
  for (size_t loop = 0; loop < numberOfIterations; ++loop)
  {
    // place ants on random points as defined in formula (9)
    initializeAnts(neighbours, gd, medianvalue, antLocations);

    // perform ant search, spread pheromone on visited data points
    antSearch(data,
	      neighbours,
	      neighbourhoodMap,
	      antLocations,
	      eigenVectors,
	      eigenValues,
	      kdTree,
	      numberOfSteps,
	      radius,
	      beta,
	      kappa,
	      p_release,
	      pheromone);
    
    // apply evaporation of pheromone as defined in formula (1)
    evaporatePheromone(pheromone, evapRate, lowerlimit, upperlimit);
    
    updateProgressBar(loop);
  }
  completeProgressBar();

  // possibly warn user
  if (lostAnts)
  {
    std::cout.clear();
    printWarning(antLocations.size() * numberOfIterations * numberOfSteps);
    if (!showProgess)
      std::cout.setstate(std::ios_base::failbit);
  }

  std::cout << "Locally Aligned Ant Technique algorithm completed\n\n";
  return pheromone;
}

#endif
