/*******************************************************************************
 * This file is part of griddle.
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

/* Methods for generating Gaussian random fields */

#ifndef GAUSSIAN_FIELD_H
#define GAUSSIAN_FIELD_H

#include <complex.h>
#include <fftw3.h>
#include <mpi.h>

#include "random.h"
#include "distributed_grid.h"

int generate_complex_grf(struct distributed_grid *dg, rng_state *state);
int generate_ngeniclike_grf(struct distributed_grid *dg, int Seed);
int enforce_hermiticity(struct distributed_grid *dg);

#endif
