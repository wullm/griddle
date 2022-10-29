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
static inline double safeNGP(const struct distributed_grid *dg, int N, int i,
                             int j, int k) {

    // if (i < dg->X0) {
    //     return dg->buffer_left[row_major_dg_buffer_left(i, j, k, dg)];
    // } else if (i >= dg->X0 + dg->NX) {
    //     return dg->buffer_right[row_major_dg_buffer_right(i, j, k, dg)];
    // } else {
    //     return dg->box[row_major_dg2(i, j, k, dg)];
    // }

    return dg->buffered_box[row_major_dg3(i, j, k, dg)];

}

/* Direct nearest grid point interpolation */
static inline double fastNGP(const struct distributed_grid *dg, int N, int i,
                             int j, int k) {
    return dg->buffered_box[row_major_dg3(i, j, k, dg)];
}

/* Direct cloud in cell interpolation */
static inline double safeCIC(const struct distributed_grid *dg, int N, int i,
                             int j, int k, double dx, double dy, double dz,
                             double tx, double ty, double tz) {

    return safeNGP(dg, N, i, j, k) * tx * ty * tz
         + safeNGP(dg, N, i, j, k+1) * tx * ty * dz
         + safeNGP(dg, N, i, j+1, k) * tx * dy * tz
         + safeNGP(dg, N, i, j+1, k+1) * tx * dy * dz
         + safeNGP(dg, N, i+1, j, k) * dx * ty * tz
         + safeNGP(dg, N, i+1, j, k+1) * dx * ty * dz
         + safeNGP(dg, N, i+1, j+1, k) * dx * dy * tz
         + safeNGP(dg, N, i+1, j+1, k+1) * dx * dy * dz;
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
void accelCIC_single(const struct distributed_grid *dg, double *x, double *a) {

    /* Physical length to grid conversion factor */
    int N = dg->N;
    double boxlen = dg->boxlen;
    double fac = N / boxlen;
    double fac_over_12 = fac / 12;

    /* Convert to float grid dimensions */
    double X = x[0] * fac;
    double Y = x[1] * fac;
    double Z = x[2] * fac;

    /* Integer grid position (floor is necessary to handle negatives) */
    int iX = floor(X);
    int iY = floor(Y);
    int iZ = floor(Z);

    /* Displacements from grid corner */
    double dx = X - iX;
    double dy = Y - iY;
    double dz = Z - iZ;
    double tx = 1.0 - dx;
    double ty = 1.0 - dy;
    double tz = 1.0 - dz;

    a[0] = 0.0;
    a[0] -= fastCIC(dg, N, iX + 2, iY, iZ, dx, dy, dz, tx, ty, tz);
    a[0] += fastCIC(dg, N, iX + 1, iY, iZ, dx, dy, dz, tx, ty, tz) * 8.;
    a[0] -= fastCIC(dg, N, iX - 1, iY, iZ, dx, dy, dz, tx, ty, tz) * 8.;
    a[0] += fastCIC(dg, N, iX - 2, iY, iZ, dx, dy, dz, tx, ty, tz);
    a[0] *= fac_over_12;

    a[1] = 0.0;
    a[1] -= fastCIC(dg, N, iX, iY + 2, iZ, dx, dy, dz, tx, ty, tz);
    a[1] += fastCIC(dg, N, iX, iY + 1, iZ, dx, dy, dz, tx, ty, tz) * 8.;
    a[1] -= fastCIC(dg, N, iX, iY - 1, iZ, dx, dy, dz, tx, ty, tz) * 8.;
    a[1] += fastCIC(dg, N, iX, iY - 2, iZ, dx, dy, dz, tx, ty, tz);
    a[1] *= fac_over_12;

    a[2] = 0.0;
    a[2] -= fastCIC(dg, N, iX, iY, iZ + 2, dx, dy, dz, tx, ty, tz);
    a[2] += fastCIC(dg, N, iX, iY, iZ + 1, dx, dy, dz, tx, ty, tz) * 8.;
    a[2] -= fastCIC(dg, N, iX, iY, iZ - 1, dx, dy, dz, tx, ty, tz) * 8.;
    a[2] += fastCIC(dg, N, iX, iY, iZ - 2, dx, dy, dz, tx, ty, tz);
    a[2] *= fac_over_12;

}


/* Compute the acceleration from the potential grid using CIC interpolation */
void accelCIC(const struct distributed_grid *dg, double *x, double *a) {

    /* Physical length to grid conversion factor */
    int N = dg->N;
    double boxlen = dg->boxlen;
    double fac = N / boxlen;
    double fac_over_12 = fac / 12;

    /* Convert to float grid dimensions */
    double X = x[0] * fac;
    double Y = x[1] * fac;
    double Z = x[2] * fac;

    /* Integer grid position (floor is necessary to handle negatives) */
    int iX = floor(X);
    int iY = floor(Y);
    int iZ = floor(Z);

    /* Displacements from grid corner */
    double dx = X - iX;
    double dy = Y - iY;
    double dz = Z - iZ;
    double tx = 1.0 - dx;
    double ty = 1.0 - dy;
    double tz = 1.0 - dz;

    a[0] = 0.0;
    a[0] -= fastCIC(dg, N, iX + 2, iY, iZ, dx, dy, dz, tx, ty, tz);
    a[0] += fastCIC(dg, N, iX + 1, iY, iZ, dx, dy, dz, tx, ty, tz) * 8.;
    a[0] -= fastCIC(dg, N, iX - 1, iY, iZ, dx, dy, dz, tx, ty, tz) * 8.;
    a[0] += fastCIC(dg, N, iX - 2, iY, iZ, dx, dy, dz, tx, ty, tz);
    a[0] *= fac_over_12;

    a[1] = 0.0;
    a[1] -= fastCIC(dg, N, iX, iY + 2, iZ, dx, dy, dz, tx, ty, tz);
    a[1] += fastCIC(dg, N, iX, iY + 1, iZ, dx, dy, dz, tx, ty, tz) * 8.;
    a[1] -= fastCIC(dg, N, iX, iY - 1, iZ, dx, dy, dz, tx, ty, tz) * 8.;
    a[1] += fastCIC(dg, N, iX, iY - 2, iZ, dx, dy, dz, tx, ty, tz);
    a[1] *= fac_over_12;

    a[2] = 0.0;
    a[2] -= fastCIC(dg, N, iX, iY, iZ + 2, dx, dy, dz, tx, ty, tz);
    a[2] += fastCIC(dg, N, iX, iY, iZ + 1, dx, dy, dz, tx, ty, tz) * 8.;
    a[2] -= fastCIC(dg, N, iX, iY, iZ - 1, dx, dy, dz, tx, ty, tz) * 8.;
    a[2] += fastCIC(dg, N, iX, iY, iZ - 2, dx, dy, dz, tx, ty, tz);
    a[2] *= fac_over_12;

}
