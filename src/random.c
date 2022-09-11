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

#include <stdlib.h>
#include <math.h>

#include "../include/random.h"

/* Generate a uniform variable on the open unit interval */
double sampleUniform(rng_state *state) {
    const uint64_t A = rand_uint64(state);
    const double RM = (double) UINT64_MAX + 1;
    return ((double) A + 0.5) / RM;
}

/* Generate standard normal variable with Box-Mueller */
double sampleNorm(rng_state *state) {
    /* Generate random integers */
    const uint64_t A = rand_uint64(state);
    const uint64_t B = rand_uint64(state);
    const double RMax = (double) UINT64_MAX + 1;

    /* Map the random integers to the open (!) unit interval */
    const double u = ((double) A + 0.5) / RMax;
    const double v = ((double) B + 0.5) / RMax;

    /* Map to two Gaussians (the second is not used - inefficient) */
    const double z0 = sqrt(-2 * log(u)) * cos(2 * M_PI * v);
    //double z1 = sqrt(-2 * log(u)) * sin(2 * M_PI * v);

    return z0;
}