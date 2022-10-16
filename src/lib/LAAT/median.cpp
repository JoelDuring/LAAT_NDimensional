#include "LAAT.ih"

size_t median(vector<unsigned int> const &data)
{
  size_t n = data.size() / 2;
  vector<unsigned int> copy(data);
  nth_element(copy.begin(), copy.begin() + n, copy.end());
  return copy[n];
}
