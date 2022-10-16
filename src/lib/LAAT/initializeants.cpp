#include "LAAT.ih"

/**
 * choose the initial points so that their neighborhood size is bigger
 * or equal to the median number of neighbors.
 *
 * @param neighbourhoods vector containing the neighboorhoods for all data
 *   points
 * @param gd vector containing the sector number for each data point
 * @param medianvalue the median number of neighbors over all data points
 * @param antLocations vector to store the initial locations in
 */
void initializeAnts(vector<unsigned int> const &neighbours,
		    vector<unsigned int> const &gd,
		    size_t medianvalue,
  		    vector<unsigned int> &antLocations)
{
  vector<bool> assigned(antLocations.size(), false);
  size_t iter = 0;
  size_t initializedAnts = 0;
  /*
    when count reaches the size of the data we assume there
    are some sectors without suitable datapoints. We then
    initialize the ants corresponding to this region with
    random suitable datapoints from other sectors.
  */
  while (initializedAnts < antLocations.size() && iter < neighbours.size())
  {
    size_t currentPos = rand() % neighbours.size();

    if (neighbours[currentPos] > medianvalue
	&& !assigned[gd[currentPos]])
    {
      antLocations[gd[currentPos]] = currentPos;
      ++initializedAnts;
      assigned[gd[currentPos]] = true;
    }

    ++iter;
  }

  if (initializedAnts != antLocations.size())
    // choose random points for uninitialized ants
    for (size_t j = 0; j < antLocations.size(); ++j)
    {
      if (!assigned[j])
	do
	  antLocations[j] = rand() % neighbours.size();
	while (neighbours[antLocations[j]] <= medianvalue);
    }
}
