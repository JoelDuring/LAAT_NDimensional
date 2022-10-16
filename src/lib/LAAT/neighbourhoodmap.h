#ifndef NEIGHBOURHOODMAP_H
#define NEIGHBOURHOODMAP_H

#include <unordered_map>
#include "utils/KDTreeVectorOfVectorsAdaptor.h"

template <size_t Dim>
using KDTree = KDTreeVectorOfVectorsAdaptor<
  std::vector<std::array<float, Dim>>, float>;

template <size_t Dim>
class NeighbourhoodMap:
  public std::unordered_map<unsigned int, std::vector<unsigned int>>
{
  std::vector<std::array<float, Dim>> const &d_data;
  std::vector<unsigned int> const &d_densities;
  KDTree<Dim> &d_kdTree;
  float d_radius;
  
public:
  NeighbourhoodMap(float memoryLimit,
		   std::vector<std::array<float, Dim>> const &data,
		   std::vector<unsigned int> const &densities,
		   KDTree<Dim> &kdTree,
		   float radius)
    :
    d_data(data),
    d_densities(densities),
    d_kdTree(kdTree),
    d_radius(radius)
    {
      if (memoryLimit == std::numeric_limits<float>::infinity())
	computeAllNeighbourhoods();
      else
	fillMap(memoryLimit);
    }

private:
  void computeAllNeighbourhoods()
    {
      // compute and store neighbourhoods
      for (size_t i = 0; i < d_data.size(); ++i)
      {
	// get neighbourhood from kd-tree
	std::vector<std::pair<size_t, float>> matches;
	size_t nMatches = d_kdTree.index->radiusSearch(&d_data[i][0],
						       d_radius * d_radius,
						       matches,
						       nanoflann::SearchParams());
	std::vector<unsigned int> neighbourhood(nMatches - 1);

	// add all neighbors but not the point itself
	for (size_t j = 1; j < nMatches; ++j)
	  neighbourhood[j - 1] = matches[j].first;

	// insert neighbourhood
	this->insert({i, neighbourhood});
      }
    }

  void fillMap(float memoryLimit)
    {
      size_t storage = memoryLimit * (1 << 30) / 4; // number of ints
      
      // array of indices 0..densities.size()
      std::vector<unsigned int> idxs(d_densities.size());
      std::iota(idxs.begin(), idxs.end(), 0);

      // sort indexes by descending density, use stable sort to avoid
      // unneccessary reordering of equal values
      std::stable_sort(idxs.begin(), idxs.end(),
		       // compare indexes based on the density
		       [this](unsigned int i, unsigned int j)
			 {
			   return d_densities[i] > d_densities[j];
			 });

      // find how many neighbourhoods we can store
      unsigned int cutoff = 0;
      size_t amount = 0;
      while (cutoff < idxs.size() && amount + d_densities[idxs[cutoff]] <= storage)
      {
	amount += d_densities[idxs[cutoff]];
	++cutoff;
      }

      // reduce to only the neighbourhoods we will store
      idxs.resize(cutoff);

      // compute and store neighbourhoods
      for (unsigned int idx : idxs)
      {
	// get neighbourhood from kd-tree
	std::vector<std::pair<size_t, float>> matches;
	size_t nMatches = d_kdTree.index->radiusSearch(&d_data[idx][0],
						       d_radius * d_radius,
						       matches,
						       nanoflann::SearchParams());
	std::vector<unsigned int> neighbourhood(nMatches - 1);

	// add all neighbors but not the point itself
	for (size_t j = 1; j < nMatches; ++j)
	  neighbourhood[j - 1] = matches[j].first;

	// insert neighbourhood
	this->insert({idx, neighbourhood});
      }
    }
};

#endif
