#ifndef ANTSEARCH_H
#define ANTSEARCH_H

/**
 * Perform ant search for all ants.
 *
 * When the function is done there is an increase of pheromone on data 
 * points dependent on how many times they were visited by ants
 *
 * This implementation uses OpenMP to perform ant search in parallel for
 * the different ants. Synchronization in the from of atomic operations
 * is used to ensure thread safety.
 *
 * @param Dim template parameter indicating the dimensionality of the data
 * @param data vector containing the data points to search on
 * @param neighbourhoods vector containing a vector of all nearest
 *   neighbours for each data point
 * @param antLocations vector containing for each ant the initial point
 *   to start searching from
 * @param eigenVectors vector containing the eigen vectors for each data
 *   point
 * @param eigenValues vector containing the eigen values for each data
 *   point
 * @param numberOfSteps number of steps to run each ant at each iteration
 * @param beta the inverse temperature as used in formula (8)
 * @param kappa tuning parameter for the relative importance of the
 *   influence of the alignment and pheromone terms as used in formula (7)
 * @param p_release amount of pheromone to release on a data point at each
 *   visit
 * @param pheromone vector containing the pheromone of each data point, will
 *   be updated by this function
 */
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
  	       std::vector<float> &pheromone)
{
#pragma omp parallel for
  for (unsigned int antLocation : antLocations)
  {
    std::vector<float> accumulatedPheromone(data.size(), 0);
    
    unsigned int current = antLocation;

    std::vector<float> V;
    std::vector<float> P;

    for (size_t i = 1; i < numberOfSteps; ++i)
    {
      /*
	From current point, select the next point from it's neighbourhood
	with probabilities as defined in formula (8).
      */
      std::array<float, Dim> const &currentPoint = data[current];

      // get neighbourhood from map if present, otherwise compute it
      std::vector<unsigned int> neighbourhood;
      if (neighbourhoodMap.count(current))
	neighbourhood = neighbourhoodMap.at(current);
      else
      {
	// get neighbourhood from kd-tree
	std::vector<std::pair<size_t, float>> matches;
	size_t nMatches = kdTree.index->radiusSearch(&data[current][0],
						     radius * radius,
						     matches,
						     nanoflann::SearchParams());
        neighbourhood.resize(nMatches - 1);

	// add all neighbors but not the point itself
	for (size_t j = 1; j < nMatches; ++j)
	  neighbourhood[j - 1] = matches[j].first;
      }
      
      std::vector<std::array<float, Dim>> relativeDistances(neighbourhood.size());

      V.resize(neighbourhood.size());
      P.resize(neighbourhood.size());

      float sumPheromone = 0;  // total pheromone in local neighbourhood
      for (size_t neighbourIdx = 0;
	   neighbourIdx < neighbourhood.size();
	   ++neighbourIdx)
      {
	std::array<float, Dim> const &neighbour = data[neighbourhood[neighbourIdx]];
	sumPheromone += pheromone[neighbourhood[neighbourIdx]];

	float distance = 0;
	for (size_t d = 0; d < Dim; ++d)
	{
          relativeDistances[neighbourIdx][d] = neighbour[d] - currentPoint[d];
	  distance +=
	    relativeDistances[neighbourIdx][d] * relativeDistances[neighbourIdx][d];
	}

	// normalize distance
	float norm = sqrt(distance);

	for (float &relDis : relativeDistances[neighbourIdx])
	  relDis /= norm;
      }

      /*
	Compute alignment between the data point and it's neighbours with
	the eigen-directions.
      */
      std::vector<std::array<float, Dim>> w(neighbourhood.size());
      // matrix multiplication
      for (size_t neighbour = 0;
	   neighbour < neighbourhood.size();
	   ++neighbour)
      {
	for (size_t d = 0; d < Dim; ++d)
	{
	  float alignment = 0;
	  for (size_t d2 = 0; d2 < Dim; ++d2)
	    alignment +=
	      relativeDistances[neighbour][d2] * eigenVectors[current][d2][d];

	  w[neighbour][d] = abs(alignment);
	}
      }

      /*
	Normalize alignment values to obtain relative weighting of the
	alignment according to formula (2).
      */
      for (std::array<float, Dim> &element : w)
      {
	float sum = std::accumulate(element.begin(), element.end(), 0.0);
        
	for (float &el : element)
	  el /= sum;
      }

      /*
	For each data point, compute the preference of moving to it's
	neigbours according to formula (4).
      */
      std::vector<float> E(neighbourhood.size());
      float sumE = 0;
      // matrix-vector multiplication
      for (size_t j = 0; j < neighbourhood.size(); ++j)
      {
	E[j] = 0;
	for (size_t d = 0; d < Dim; ++d)
	E[j] += w[j][d] * eigenValues[current][d];

	sumE += E[j];
      }

      float sumV = 0.0;
      /*
	calculate movement preference from the current point to all
	of it's neighbours using formula (7).
      */    
      for (size_t j = 0; j < neighbourhood.size(); ++j)
      {
	unsigned int neighbour = neighbourhood[j];
	if (neighbours[neighbour])  // neighbour is active
	{
	  // calculate normalized pheromone as defined in formula (6).
	  float normalizedPheromone =
	    pheromone[neighbour] / sumPheromone;

	  /*
	    Normalize movement preferences within the neighbourhood to
	    obtain the relative preference according to formula (5).
	  */
	  float preference = E[j] / sumE;

	  // formula (7)
	  V[j] = exp(
	    beta * ((1 - kappa) * normalizedPheromone +
		    kappa * preference)
	    );
	  sumV += V[j];
	}
	else
	  V[j] = 0;
      }

      /*
	Calculate jump probabilities to all of the current point's
	neighbours, store as cummulative probabilities for easy selection.
      */
      float cummulativeProbability = 0.0;
      size_t itemp = 0;
      float randnum = (float) rand() / (RAND_MAX);
      while ((itemp + 1) < neighbourhood.size())
      {
	cummulativeProbability += V[itemp] / sumV;
	if (cummulativeProbability > randnum)
	  break;
	
	++itemp;
      }

      current = neighbourhood[itemp];

      /*
	if there are no active neighbours, update counter for sake of 
	warning the user
      */
      if (!neighbours[current])
      {
        ++lostAnts;
	break;
      }

      // keep track of pheromone to be added
      accumulatedPheromone[current] += p_release;
    }

    /*
      Update pheromone on all visited points as defined in formula (10).
    */
    for (size_t idx = 0; idx < data.size(); ++idx)
      if (accumulatedPheromone[idx])
      {
	/*
	  ensure that no two threads will ever update the same value at
	  the same time
	*/
      #pragma omp atomic
        pheromone[idx] += accumulatedPheromone[idx];
      }
  }
}


#endif
