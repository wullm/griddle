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

double gridNGP(const struct distributed_grid *dg, int N, double boxlen,
               double x, double y, double z);
double gridCIC(const struct distributed_grid *dg, int N, double boxlen,
               double x, double y, double z);
double gridInterp(const struct distributed_grid *dg, int N, double boxlen,
                  double x, double y, double z, int order);
void accelNGP(const struct distributed_grid *dg, int N, double boxlen,
              double *x, double *a);
void accelCIC(const struct distributed_grid *dg, int N, double boxlen,
              double *x, double *a);
void accelInterp(const struct distributed_grid *dg, int N, double boxlen,
                 double *x, double *a, int order);
#endif
