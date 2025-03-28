/* 
 * This function originated from https://github.com/LCAV/pyroomacoustics and is modified by Fraunhofer
 * The code was obtained under the MIT license, which is distributed with this software
 * Copyright 2024 Fraunhofer IIS
*/ 
#ifndef __RIR_BUILDER_HPP__
#define __RIR_BUILDER_HPP__

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void rir_builder(py::buffer rir, const py::buffer &time, const py::buffer &alpha,
                 const py::buffer &visibility, int fs, size_t fdl,
                 size_t lut_gran, size_t num_threads);

void delay_sum(const py::buffer irs, const py::buffer delays, py::buffer output,
               size_t num_threads);

void fractional_delay(py::buffer out, const py::buffer time, size_t lut_gran,
                      size_t num_threads);

#endif  // __RIR_BUILDER_HPP__
