/*******************************************************************************
 * This file is part of Sedulus.
 * Copyright (c) 2021 Willem Elbers (whe@willemelbers.com)
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

#ifndef MESH_GRAVITY_H
#define MESH_GRAVITY_H

#include "../include/distributed_grid.h"

static inline long int row_major_index(int i, int j, int k, long int N, long int Nz) {
    return i*N*Nz + j*Nz + k;
}

/* Agnostic acceleration through differentiation of the potential grid */
void accelCIC(const struct distributed_grid *dg, const double x[3], double a[3]);

/* Implementations of differentiation using different order schemes */
void accelCIC_4th(const GridFloatType *box, const double x[3], double a[3],
                  const long int N, const int X0, const int buffer_width,
                  const long int Nz, const double cell_fac);
void accelCIC_2nd(const GridFloatType *box, const double x[3], double a[3],
                  const long int N, const int X0, const int buffer_width,
                  const long int Nz, const double cell_fac);
void accelCIC_1st(const GridFloatType *box, const double x[3], double a[3],
                  const long int N, const int X0, const int buffer_width,
                  const long int Nz, const double cell_fac);

#endif
