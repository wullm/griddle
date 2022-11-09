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

#include <math.h>
#include <stdlib.h>
#include "../include/mesh_grav.h"
#include "../include/fft.h"

/* Direct nearest grid point interpolation */
static inline double fastNGP(const struct distributed_grid *dg, int N, int i,
                             int j, int k) {
    return *point_row_major_dg_buffered(i, j, k, dg);
}

/* Direct nearest grid point interpolation (without bounds checking) */
static inline GridFloatType nowrapNGP(const struct distributed_grid *dg, int i, int j, int k) {
    return *point_row_major_dg_buffered_nobounds(i, j, k, dg);
}

/* Direct cloud in cell interpolation */
static inline double fastCIC(const struct distributed_grid *dg, int N, int i,
                             int j, int k, double dx, double dy, double dz,
                             double tx, double ty, double tz) {

    return fastNGP(dg, N, i, j, k) * tx * ty * tz
         + fastNGP(dg, N, i, j, k+1) * tx * ty * dz
         + fastNGP(dg, N, i, j+1, k) * tx * dy * tz
         + fastNGP(dg, N, i, j+1, k+1) * tx * dy * dz
         + fastNGP(dg, N, i+1, j, k) * dx * ty * tz
         + fastNGP(dg, N, i+1, j, k+1) * dx * ty * dz
         + fastNGP(dg, N, i+1, j+1, k) * dx * dy * tz
         + fastNGP(dg, N, i+1, j+1, k+1) * dx * dy * dz;
}

/* Compute the acceleration from the potential grid using CIC interpolation */
void accelCIC(const struct distributed_grid *dg, const double x[3], double a[3]) {

    /* Physical length to grid conversion factor */
    int N = dg->N;
    double boxlen = dg->boxlen;
    double fac = N / boxlen;

    GridFloatType *box = dg->buffered_box;
    int Nz = dg->Nz;
    int buffer_width = dg->buffer_width;
    int X0 = dg->X0;

    /* Use the fourth order differentiation scheme by default */
    accelCIC_4th(box, x, a, N, X0, buffer_width, Nz, fac);
}
