/*******************************************************************************
 * This file is part of Sedulus.
 * Copyright (c) 2022 Willem Elbers (whe@willemelbers.com)
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/

#ifndef FFT_TYPES_H
#define FFT_TYPES_H

#include "config.h"

#ifdef SINGLE_PRECISION_FFTW
typedef float GridFloatType;
typedef fftwf_complex GridComplexType;
typedef fftwf_plan FourierPlanType;
#define MPI_GRID_TYPE MPI_FLOAT
#define MPI_COMPLEX_GRID_TYPE MPI_COMPLEX
#define H5T_GRID_TYPE H5T_NATIVE_FLOAT
#else
typedef double GridFloatType;
typedef fftw_complex GridComplexType;
typedef fftw_plan FourierPlanType;
#define MPI_GRID_TYPE MPI_DOUBLE
#define MPI_COMPLEX_GRID_TYPE MPI_DOUBLE_COMPLEX
#define H5T_GRID_TYPE H5T_NATIVE_DOUBLE
#endif

#ifdef MEASURED_FFTW_PLANS
#define PREPARE_FLAG FFTW_MEASURE
#else
#define PREPARE_FLAG FFTW_ESTIMATE
#endif

#endif
