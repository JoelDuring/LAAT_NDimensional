#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <iostream>
#include <fstream>
#include "LAAT/LAAT.h"
#include "MBMS/MBMS.h"

namespace py = pybind11;
using namespace py::literals;

/**
 * Binding function to use the LAAT algorithm from Python.
 *
 * The function expects a Numpy array containing the data, and returns a Numpy
 * array containing the pheromone resulting from LAAT.
 */
template <size_t Dim>
py::array LAAT(py::array_t<float> in,
	       std::array<size_t, Dim> ants,
	       size_t numberOfIterations,
	       size_t numberOfSteps,
	       size_t threshold,
	       float memoryLimit,
	       float radius,
	       float beta,
	       float kappa,
	       float p_release,
	       float evapRate,
	       float lowerlimit,
	       float upperlimit,
	       bool showProgress)
{
  auto buf = in.request();
  float *npData = static_cast<float *>(buf.ptr);
  size_t size = buf.shape[0];
  std::vector<std::array<float, Dim>> data(size, std::array<float, Dim>());

  for (size_t i = 0; i < size; ++i)
    for (size_t j = 0; j < 3; ++j)
      data[i][j] = npData[i + j * size];

  py::array ret = py::cast(LocallyAlignedAntTechnique(data,
						      ants,
						      numberOfIterations,
						      numberOfSteps,
						      threshold,
						      memoryLimit,
						      radius,
						      beta,
						      kappa,
						      p_release,
						      evapRate,
						      lowerlimit,
						      upperlimit,
						      showProgress));
  return ret;
}

/**
 * Binding function to use the MBMS algorithm from Python.
 *
 * The function expects a Numpy array containing the data, and returns a Numpy
 * array containing updated data after MBMS has been applied.
 */
py::array MBMS(py::array_t<float> in,
	       size_t dim,
	       size_t iter,
	       float radius,
	       float sigma,
	       size_t k)
{
  auto buf = in.request();
  float *npData = static_cast<float *>(buf.ptr);

  size_t size = buf.shape[0];
  std::vector<std::vector<float>> data(size, std::vector<float>(dim));

  for (size_t i = 0; i < size; ++i)
    for (size_t j = 0; j < dim; ++j)
      data[i][j] = npData[i * dim + j];

  manifoldBlurringMeanShift(data, dim, iter, radius, sigma, k);
  
  return py::array(py::cast(data));
}

PYBIND11_MODULE(laat, m)
{
  m.doc() = "Cosmic Web module for Python. Contains the LAAT and MBMS functions.";

    // definitions of LAAT for 3- up to 12-dimensional data
  m.def("LAAT", &LAAT<3>, "Locally Aligned Ant Technique algorithm",
	"in"_a,
	"ants"_a = std::array<size_t, 3>{8, 5, 5},
	"numberOfIterations"_a = 100,
	"numberOfSteps"_a = 12000,
	"threshold"_a = 3,
	"memoryLimit"_a = std::numeric_limits<float>::infinity(),
	"radius"_a = 0.5,
	"beta"_a = 10.0,
	"kappa"_a = 0.8,
	"p_release"_a = 0.05,
	"evapRate"_a = 0.05,
	"lowerlimit"_a = 0.0001,
	"upperlimit"_a = 10,
        "showProgress"_a = true);

  m.def("LAAT4", &LAAT<4>, "Locally Aligned Ant Technique algorithm",
	"in"_a,
	"ants"_a = std::array<size_t, 4>{5, 5, 5, 5},
	"numberOfIterations"_a = 100,
	"numberOfSteps"_a = 12000,
	"threshold"_a = 4,
	"memoryLimit"_a = std::numeric_limits<float>::infinity(),
	"radius"_a = 0.5,
	"beta"_a = 10.0,
	"kappa"_a = 0.8,
	"p_release"_a = 0.05,
	"evapRate"_a = 0.05,
	"lowerlimit"_a = 0.0001,
	"upperlimit"_a = 10,
        "showProgress"_a = true);	

  m.def("LAAT5", &LAAT<5>, "Locally Aligned Ant Technique algorithm",
	"in"_a,
	"ants"_a = std::array<size_t, 5>{5, 5, 5, 5, 5},
	"numberOfIterations"_a = 100,
	"numberOfSteps"_a = 12000,
	"threshold"_a = 5,
	"memoryLimit"_a = std::numeric_limits<float>::infinity(),
	"radius"_a = 0.5,
	"beta"_a = 10.0,
	"kappa"_a = 0.8,
	"p_release"_a = 0.05,
	"evapRate"_a = 0.05,
	"lowerlimit"_a = 0.0001,
	"upperlimit"_a = 10,
        "showProgress"_a = true);	

  m.def("LAAT6", &LAAT<6>, "Locally Aligned Ant Technique algorithm",
	"in"_a,
	"ants"_a = std::array<size_t, 6>{5, 5, 5, 5, 5, 5},
	"numberOfIterations"_a = 100,
	"numberOfSteps"_a = 12000,
	"threshold"_a = 6,
	"memoryLimit"_a = std::numeric_limits<float>::infinity(),
	"radius"_a = 0.5,
	"beta"_a = 10.0,
	"kappa"_a = 0.8,
	"p_release"_a = 0.05,
	"evapRate"_a = 0.05,
	"lowerlimit"_a = 0.0001,
	"upperlimit"_a = 10,
        "showProgress"_a = true);	

  m.def("LAAT7", &LAAT<7>, "Locally Aligned Ant Technique algorithm",
	"in"_a,
	"ants"_a = std::array<size_t, 7>{5, 5, 5, 5, 5, 5, 5},
	"numberOfIterations"_a = 100,
	"numberOfSteps"_a = 12000,
	"threshold"_a = 7,
	"memoryLimit"_a = std::numeric_limits<float>::infinity(),
	"radius"_a = 0.5,
	"beta"_a = 10.0,
	"kappa"_a = 0.8,
	"p_release"_a = 0.05,
	"evapRate"_a = 0.05,
	"lowerlimit"_a = 0.0001,
	"upperlimit"_a = 10,
        "showProgress"_a = true);	

  m.def("LAAT8", &LAAT<8>, "Locally Aligned Ant Technique algorithm",
	"in"_a,
	"ants"_a = std::array<size_t, 8>{5, 5, 5, 5, 5, 5, 5, 5},
	"numberOfIterations"_a = 100,
	"numberOfSteps"_a = 12000,
	"threshold"_a = 8,
	"memoryLimit"_a = std::numeric_limits<float>::infinity(),
	"radius"_a = 0.5,
	"beta"_a = 10.0,
	"kappa"_a = 0.8,
	"p_release"_a = 0.05,
	"evapRate"_a = 0.05,
	"lowerlimit"_a = 0.0001,
	"upperlimit"_a = 10,
        "showProgress"_a = true);	

  m.def("LAAT9", &LAAT<9>, "Locally Aligned Ant Technique algorithm",
	"in"_a,
	"ants"_a = std::array<size_t, 9>{5, 5, 5, 5, 5, 5, 5, 5, 5},
	"numberOfIterations"_a = 100,
	"numberOfSteps"_a = 12000,
	"threshold"_a = 9,
	"memoryLimit"_a = std::numeric_limits<float>::infinity(),
	"radius"_a = 0.5,
	"beta"_a = 10.0,
	"kappa"_a = 0.8,
	"p_release"_a = 0.05,
	"evapRate"_a = 0.05,
	"lowerlimit"_a = 0.0001,
	"upperlimit"_a = 10,
        "showProgress"_a = true);	

  m.def("LAAT10", &LAAT<10>, "Locally Aligned Ant Technique algorithm",
	"in"_a,
	"ants"_a = std::array<size_t, 10>{5, 5, 5, 5, 5, 5, 5, 5, 5, 5},
	"numberOfIterations"_a = 100,
	"numberOfSteps"_a = 12000,
	"threshold"_a = 10,
	"memoryLimit"_a = std::numeric_limits<float>::infinity(),
	"radius"_a = 0.5,
	"beta"_a = 10.0,
	"kappa"_a = 0.8,
	"p_release"_a = 0.05,
	"evapRate"_a = 0.05,
	"lowerlimit"_a = 0.0001,
	"upperlimit"_a = 10,
        "showProgress"_a = true);	

  m.def("LAAT11", &LAAT<11>, "Locally Aligned Ant Technique algorithm",
	"in"_a,
	"ants"_a = std::array<size_t, 11>{5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5},
	"numberOfIterations"_a = 100,
	"numberOfSteps"_a = 12000,
	"threshold"_a = 11,
	"memoryLimit"_a = std::numeric_limits<float>::infinity(),
	"radius"_a = 0.5,
	"beta"_a = 10.0,
	"kappa"_a = 0.8,
	"p_release"_a = 0.05,
	"evapRate"_a = 0.05,
	"lowerlimit"_a = 0.0001,
	"upperlimit"_a = 10,
        "showProgress"_a = true);	

  m.def("LAAT12", &LAAT<12>, "Locally Aligned Ant Technique algorithm",
	"in"_a,
	"ants"_a = std::array<size_t, 12>{5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5},
	"numberOfIterations"_a = 100,
	"numberOfSteps"_a = 12000,
	"threshold"_a = 12,
	"memoryLimit"_a = std::numeric_limits<float>::infinity(),
	"radius"_a = 0.5,
	"beta"_a = 10.0,
	"kappa"_a = 0.8,
	"p_release"_a = 0.05,
	"evapRate"_a = 0.05,
	"lowerlimit"_a = 0.0001,
	"upperlimit"_a = 10,
        "showProgress"_a = true);	

  m.def("LAAT13", &LAAT<13>, "Locally Aligned Ant Technique algorithm",
	"in"_a,
	"ants"_a = std::array<size_t, 13>{5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5},
	"numberOfIterations"_a = 100,
	"numberOfSteps"_a = 12000,
	"threshold"_a = 13,
	"memoryLimit"_a = std::numeric_limits<float>::infinity(),
	"radius"_a = 0.5,
	"beta"_a = 10.0,
	"kappa"_a = 0.8,
	"p_release"_a = 0.05,
	"evapRate"_a = 0.05,
	"lowerlimit"_a = 0.0001,
	"upperlimit"_a = 10,
        "showProgress"_a = true);
  
  m.def("LAAT14", &LAAT<14>, "Locally Aligned Ant Technique algorithm",
	"in"_a,
	"ants"_a = std::array<size_t, 14>{5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5},
	"numberOfIterations"_a = 100,
	"numberOfSteps"_a = 12000,
	"threshold"_a = 14,
	"memoryLimit"_a = std::numeric_limits<float>::infinity(),
	"radius"_a = 0.5,
	"beta"_a = 10.0,
	"kappa"_a = 0.8,
	"p_release"_a = 0.05,
	"evapRate"_a = 0.05,
	"lowerlimit"_a = 0.0001,
	"upperlimit"_a = 10,
        "showProgress"_a = true);
  
  m.def("LAAT15", &LAAT<15>, "Locally Aligned Ant Technique algorithm",
	"in"_a,
	"ants"_a = std::array<size_t, 15>{5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5},
	"numberOfIterations"_a = 100,
	"numberOfSteps"_a = 12000,
	"threshold"_a = 15,
	"memoryLimit"_a = std::numeric_limits<float>::infinity(),
	"radius"_a = 0.5,
	"beta"_a = 10.0,
	"kappa"_a = 0.8,
	"p_release"_a = 0.05,
	"evapRate"_a = 0.05,
	"lowerlimit"_a = 0.0001,
	"upperlimit"_a = 10,
        "showProgress"_a = true);	

  m.def("MBMS", &MBMS, "Manifold Blurring Mean Shift algorithm",
	"in"_a,
	"dim"_a = 3,
	"iter"_a = 10,
	"radius"_a = 3,
	"sigma"_a = 1.5,
	"k"_a = 10);
}
