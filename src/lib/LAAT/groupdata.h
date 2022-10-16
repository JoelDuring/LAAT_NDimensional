#ifndef GROUPDATA_H
#define GROUPDATA_H

/**
 * Group the data into one sector for each ant, group each data point
 * into it's corresponding sector.
 *
 * @param Dim template parameter indicating the dimensionality of the data
 * @param data vector containing the data to group
 * @param ants an std::array conforming to the dimensionality of the data that
 *   indicates how many ants should be distributed along each dimension. The
 *   total number of ants will be equal to the product of the elements of this
 *   array
 * @return vector containing the sector number for each data point
 */
template <size_t Dim>
std::vector<unsigned int> groupdata(std::vector<std::array<float, Dim>> const &data,
				    std::array<size_t, Dim> const &ants)
{
  // compute extremes of the data
  std::vector<float> mins(Dim, std::numeric_limits<float>::infinity());
  std::vector<float> maxs(Dim, -std::numeric_limits<float>::infinity());

  for (size_t i = 0; i < data.size(); ++i)
    for (size_t d = 0; d < Dim; ++d)
    {
      if (data[i][d] > maxs[d])
	maxs[d] = data[i][d];
      if (data[i][d] < mins[d])
	mins[d] = data[i][d];
    }

  // create limits to separate sectors
  std::vector<std::vector<float>> lims(Dim);
  for (size_t d = 0; d < Dim; ++d)
  {
    lims[d].assign(ants[d], 0);
    for (size_t i = 0; i < ants[d]; ++i)
      lims[d][i] = mins[d] + (i + 1) * (maxs[d] - mins[d]) / ants[d];
  }
  
  /*
    Because of floating point percision several points might be bigger than 
    the highest threshold, Thus to make sure all points will belong to at 
    least one group we increase the final limit by an epsilon
  */
  for (size_t d = 0; d < Dim; ++d)
    lims[d][ants[d] - 1] += 0.0001;

  // assign sector number to each data point
  std::vector<unsigned int> groupedData(data.size());
  for (size_t i = 0; i < data.size(); ++i)
  {
    size_t antsProduct = 1;
    for (size_t d = 0; d < Dim; ++d)
    {
      for (size_t j = 0; j < ants[d]; ++j)
	if (data[i][d] < lims[d][j])
	{
	  groupedData[i] = groupedData[i] + j * antsProduct;
	  break;
	}
      antsProduct *= ants[d];
    }
  }
  return groupedData;
}

#endif
