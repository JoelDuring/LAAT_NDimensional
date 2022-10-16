#ifndef PREPROCESS_H
#define PREPROCESS_H

#include <iostream>

/**
 * Perform preprocessing steps as defined in Algorithm 1.
 *
 * When the function is done relative preferences have been calculated for
 * each data point and it's neighbours.
 *
 * @param Dim template parameter indicating the dimensionality of the data
 * @param data vector containing the data points to preprocess
 * @param radius the radius of the neighbourhood for each data point
 * @param threshold amount of neighbours a data point should at least
 *   have to be included in the search
 * @param neighbours vector to store the neighbourhood sizes in
 * @param eigenVectors vector to store the eigen vectors in
 * @param eigenValues vector to store the eigen values in
 * @return median neighbourhood size of all data points
 */
template <size_t Dim>
size_t preprocess(std::vector<std::array<float, Dim>> const &data,
		float const radius,
		float const threshold,
		std::vector<unsigned int> &neighbours,
		std::vector<std::vector<std::array<float, Dim>>> &eigenVectors,
		std::vector<std::array<float, Dim>> &eigenValues,
		KDTree<Dim> &kdTree)
{
  /*
    Compute Eigen-vectors and -values. Normalize eigen-values according
    to formula (3). Find neighbourhoods and store number of neighbours.
  */
  neighbours.resize(data.size());
  eigenVectors.resize(data.size());
  eigenValues.resize(data.size());
  for (size_t i = 0; i < data.size(); ++i)
  {
    // find neighbourhood
    std::vector<std::pair<size_t, float>> matches;
    size_t nMatches = kdTree.index->radiusSearch(&data[i][0],
						 radius * radius,
						 matches,
						 nanoflann::SearchParams());
    std::vector<unsigned int> neighbourhood(nMatches - 1);
    
    // add all neighbours except the point itself
    for (size_t j = 1; j < nMatches; ++j)
      neighbourhood[j - 1] = matches[j].first;

    // store number of neighbours
    neighbours[i] = nMatches - 1;

    if (neighbours[i] >= threshold)
    {
      Eigen::MatrixXf X(neighbourhood.size(), Dim);
      Eigen::RowVectorXf meanvec(Dim);

      for (size_t j = 0; j < neighbourhood.size(); ++j)
	for (size_t d = 0; d < Dim; ++d)
	  X(j, d) = data[neighbourhood[j]][d];

      for (size_t d = 0; d < Dim; ++d)
	meanvec(d) = X.middleCols<1>(d).mean();

      X = X.rowwise() - meanvec;
      meanvec.resize(0);  // destructor

      Eigen::BDCSVD<Eigen::MatrixXf> svd(X, Eigen::ComputeThinV);

      X.resize(0,0); // this is the destructor
      Eigen::RowVectorXf svalue(Dim);
      svalue = svd.singularValues();

      Eigen::Matrix<float, Dim, Dim> svector;
      svector = svd.matrixV();          // eigen vectors are in columns
      svalue = svalue.array().pow(2);   // second power of singular values
      svalue = svalue.array() / svalue.sum();  // normalize eigen values

      for (size_t d = 0; d < Dim; ++d)
	eigenValues[i][d] = svalue[d];
      svalue.resize(0);

      eigenVectors[i].resize(Dim, std::array<float, Dim>());
      for (size_t j = 0; j < Dim; ++j)
	for (size_t k = 0; k < Dim; ++k)
	  eigenVectors[i][j][k] = svector(j, k);
    }
  }
  size_t medianvalue = median(neighbours);

  // signal empty neighbourhoods for points with less neighbours than the
  // threshold
  for (unsigned int &n : neighbours)
    if (n < threshold)
      n = 0;

  return medianvalue;
}

#endif
