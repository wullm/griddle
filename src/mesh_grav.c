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

static inline long int row_major_index(int i, int j, int k, int N, int Nz) {
    return i*N*Nz + j*Nz + k;
}

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
/* Uses a fourth-order accurate central difference scheme */
void accelCIC(const struct distributed_grid *dg, double *x, double *a) {

    /* Physical length to grid conversion factor */
    int N = dg->N;
    double boxlen = dg->boxlen;
    double fac = N / boxlen;
    double fac_over_12 = fac / 12;

    /* Coordinates are mapped to [0, N] */
    double X = x[0];
    double Y = x[1];
    double Z = x[2];

    /* Integer grid position (floor not needed, as wrapping is ensured) */
    int iX = X;
    int iY = Y;
    int iZ = Z;

    /* Displacements from grid corner */
    double dx = X - iX;
    double dy = Y - iY;
    double dz = Z - iZ;
    double tx = 1.0 - dx;
    double ty = 1.0 - dy;
    double tz = 1.0 - dz;

    /* Products of fractional displacements from cell corners */
    double ttt = tx * ty * tz;
    double ttd = tx * ty * dz;
    double dtt = dx * ty * tz;
    double dtd = dx * ty * dz;
    double tdt = tx * dy * tz;
    double tdd = tx * dy * dz;
    double ddt = dx * dy * tz;
    double ddd = dx * dy * dz;

    /* Wrap the integer coordinates (not necessary for x) */
    int iX0 = iX - 2;
    int iY0 = wrap(iY - 2, N);
    int iZ0 = wrap(iZ - 2, N);
    int iX1 = iX - 1;
    int iY1 = wrap(iY - 1, N);
    int iZ1 = wrap(iZ - 1, N);
    int iX3 = iX + 1;
    int iY3 = wrap(iY + 1, N);
    int iZ3 = wrap(iZ + 1, N);
    int iX4 = iX + 2;
    int iY4 = wrap(iY + 2, N);
    int iZ4 = wrap(iZ + 2, N);
    int iX5 = iX + 3;
    int iY5 = wrap(iY + 3, N);
    int iZ5 = wrap(iZ + 3, N);

    /* Retrieve the values necessary for the finite difference scheme */
    double val_202 = nowrapNGP(dg, iX, iY0, iZ);
    double val_313 = nowrapNGP(dg, iX3, iY1, iZ3);
    double val_023 = nowrapNGP(dg, iX0, iY, iZ3);
    double val_332 = nowrapNGP(dg, iX3, iY3, iZ);
    double val_212 = nowrapNGP(dg, iX, iY1, iZ);
    double val_122 = nowrapNGP(dg, iX1, iY, iZ);
    double val_222 = nowrapNGP(dg, iX, iY, iZ);
    double val_221 = nowrapNGP(dg, iX, iY, iZ1);
    double val_232 = nowrapNGP(dg, iX, iY3, iZ);
    double val_343 = nowrapNGP(dg, iX3, iY4, iZ3);
    double val_333 = nowrapNGP(dg, iX3, iY3, iZ3);
    double val_225 = nowrapNGP(dg, iX, iY, iZ5);
    double val_235 = nowrapNGP(dg, iX, iY3, iZ5);
    double val_352 = nowrapNGP(dg, iX3, iY5, iZ);
    double val_132 = nowrapNGP(dg, iX1, iY3, iZ);
    double val_523 = nowrapNGP(dg, iX5, iY, iZ3);
    double val_253 = nowrapNGP(dg, iX, iY5, iZ3);
    double val_233 = nowrapNGP(dg, iX, iY3, iZ3);
    double val_303 = nowrapNGP(dg, iX3, iY0, iZ3);
    double val_022 = nowrapNGP(dg, iX0, iY, iZ);
    double val_423 = nowrapNGP(dg, iX4, iY, iZ3);
    double val_213 = nowrapNGP(dg, iX, iY1, iZ3);
    double val_325 = nowrapNGP(dg, iX3, iY, iZ5);
    double val_252 = nowrapNGP(dg, iX, iY5, iZ);
    double val_323 = nowrapNGP(dg, iX3, iY, iZ3);
    double val_342 = nowrapNGP(dg, iX3, iY4, iZ);
    double val_242 = nowrapNGP(dg, iX, iY4, iZ);
    double val_230 = nowrapNGP(dg, iX, iY3, iZ0);
    double val_234 = nowrapNGP(dg, iX, iY3, iZ4);
    double val_243 = nowrapNGP(dg, iX, iY4, iZ3);
    double val_312 = nowrapNGP(dg, iX3, iY1, iZ);
    double val_033 = nowrapNGP(dg, iX0, iY3, iZ3);
    double val_220 = nowrapNGP(dg, iX, iY, iZ0);
    double val_422 = nowrapNGP(dg, iX4, iY, iZ);
    double val_324 = nowrapNGP(dg, iX3, iY, iZ4);
    double val_231 = nowrapNGP(dg, iX, iY3, iZ1);
    double val_133 = nowrapNGP(dg, iX1, iY3, iZ3);
    double val_320 = nowrapNGP(dg, iX3, iY, iZ0);
    double val_224 = nowrapNGP(dg, iX, iY, iZ4);
    double val_432 = nowrapNGP(dg, iX4, iY3, iZ);
    double val_203 = nowrapNGP(dg, iX, iY0, iZ3);
    double val_433 = nowrapNGP(dg, iX4, iY3, iZ3);
    double val_522 = nowrapNGP(dg, iX5, iY, iZ);
    double val_533 = nowrapNGP(dg, iX5, iY3, iZ3);
    double val_334 = nowrapNGP(dg, iX3, iY3, iZ4);
    double val_123 = nowrapNGP(dg, iX1, iY, iZ3);
    double val_321 = nowrapNGP(dg, iX3, iY, iZ1);
    double val_302 = nowrapNGP(dg, iX3, iY0, iZ);
    double val_322 = nowrapNGP(dg, iX3, iY, iZ);
    double val_223 = nowrapNGP(dg, iX, iY, iZ3);
    double val_330 = nowrapNGP(dg, iX3, iY3, iZ0);
    double val_532 = nowrapNGP(dg, iX5, iY3, iZ);
    double val_353 = nowrapNGP(dg, iX3, iY5, iZ3);
    double val_032 = nowrapNGP(dg, iX0, iY3, iZ);
    double val_331 = nowrapNGP(dg, iX3, iY3, iZ1);
    double val_335 = nowrapNGP(dg, iX3, iY3, iZ5);

    /* Compute the finite difference along the x-axis */
    a[0] -= val_422 * ttt;
    a[0] -= val_423 * ttd;
    a[0] -= val_522 * dtt;
    a[0] -= val_523 * dtd;
    a[0] -= val_432 * tdt;
    a[0] -= val_433 * tdd;
    a[0] -= val_532 * ddt;
    a[0] -= val_533 * ddd;
    a[0] += val_322 * ttt * 8.0;
    a[0] += val_323 * ttd * 8.0;
    a[0] += val_422 * dtt * 8.0;
    a[0] += val_423 * dtd * 8.0;
    a[0] += val_332 * tdt * 8.0;
    a[0] += val_333 * tdd * 8.0;
    a[0] += val_432 * ddt * 8.0;
    a[0] += val_433 * ddd * 8.0;
    a[0] -= val_122 * ttt * 8.0;
    a[0] -= val_123 * ttd * 8.0;
    a[0] -= val_222 * dtt * 8.0;
    a[0] -= val_223 * dtd * 8.0;
    a[0] -= val_132 * tdt * 8.0;
    a[0] -= val_133 * tdd * 8.0;
    a[0] -= val_232 * ddt * 8.0;
    a[0] -= val_233 * ddd * 8.0;
    a[0] += val_022 * ttt;
    a[0] += val_023 * ttd;
    a[0] += val_122 * dtt;
    a[0] += val_123 * dtd;
    a[0] += val_032 * tdt;
    a[0] += val_033 * tdd;
    a[0] += val_132 * ddt;
    a[0] += val_133 * ddd;

    /* Compute the finite difference along the y-axis */
    a[1] -= val_242 * ttt;
    a[1] -= val_243 * ttd;
    a[1] -= val_342 * dtt;
    a[1] -= val_343 * dtd;
    a[1] -= val_252 * tdt;
    a[1] -= val_253 * tdd;
    a[1] -= val_352 * ddt;
    a[1] -= val_353 * ddd;
    a[1] += val_232 * ttt * 8.0;
    a[1] += val_233 * ttd * 8.0;
    a[1] += val_332 * dtt * 8.0;
    a[1] += val_333 * dtd * 8.0;
    a[1] += val_242 * tdt * 8.0;
    a[1] += val_243 * tdd * 8.0;
    a[1] += val_342 * ddt * 8.0;
    a[1] += val_343 * ddd * 8.0;
    a[1] -= val_212 * ttt * 8.0;
    a[1] -= val_213 * ttd * 8.0;
    a[1] -= val_312 * dtt * 8.0;
    a[1] -= val_313 * dtd * 8.0;
    a[1] -= val_222 * tdt * 8.0;
    a[1] -= val_223 * tdd * 8.0;
    a[1] -= val_322 * ddt * 8.0;
    a[1] -= val_323 * ddd * 8.0;
    a[1] += val_202 * ttt;
    a[1] += val_203 * ttd;
    a[1] += val_302 * dtt;
    a[1] += val_303 * dtd;
    a[1] += val_212 * tdt;
    a[1] += val_213 * tdd;
    a[1] += val_312 * ddt;
    a[1] += val_313 * ddd;

    /* Compute the finite difference along the z-axis */
    a[2] -= val_224 * ttt;
    a[2] -= val_225 * ttd;
    a[2] -= val_324 * dtt;
    a[2] -= val_325 * dtd;
    a[2] -= val_234 * tdt;
    a[2] -= val_235 * tdd;
    a[2] -= val_334 * ddt;
    a[2] -= val_335 * ddd;
    a[2] += val_223 * ttt * 8.0;
    a[2] += val_224 * ttd * 8.0;
    a[2] += val_323 * dtt * 8.0;
    a[2] += val_324 * dtd * 8.0;
    a[2] += val_233 * tdt * 8.0;
    a[2] += val_234 * tdd * 8.0;
    a[2] += val_333 * ddt * 8.0;
    a[2] += val_334 * ddd * 8.0;
    a[2] -= val_221 * ttt * 8.0;
    a[2] -= val_222 * ttd * 8.0;
    a[2] -= val_321 * dtt * 8.0;
    a[2] -= val_322 * dtd * 8.0;
    a[2] -= val_231 * tdt * 8.0;
    a[2] -= val_232 * tdd * 8.0;
    a[2] -= val_331 * ddt * 8.0;
    a[2] -= val_332 * ddd * 8.0;
    a[2] += val_220 * ttt;
    a[2] += val_221 * ttd;
    a[2] += val_320 * dtt;
    a[2] += val_321 * dtd;
    a[2] += val_230 * tdt;
    a[2] += val_231 * tdd;
    a[2] += val_330 * ddt;
    a[2] += val_331 * ddd;

    a[0] *= fac_over_12;
    a[1] *= fac_over_12;
    a[2] *= fac_over_12;

    // a[0] = 0.0;
    // a[0] -= fastCIC(dg, N, iX + 2, iY, iZ, dx, dy, dz, tx, ty, tz);
    // a[0] += fastCIC(dg, N, iX + 1, iY, iZ, dx, dy, dz, tx, ty, tz) * 8.;
    // a[0] -= fastCIC(dg, N, iX - 1, iY, iZ, dx, dy, dz, tx, ty, tz) * 8.;
    // a[0] += fastCIC(dg, N, iX - 2, iY, iZ, dx, dy, dz, tx, ty, tz);
    // a[0] *= fac_over_12;
    //
    // a[1] = 0.0;
    // a[1] -= fastCIC(dg, N, iX, iY + 2, iZ, dx, dy, dz, tx, ty, tz);
    // a[1] += fastCIC(dg, N, iX, iY + 1, iZ, dx, dy, dz, tx, ty, tz) * 8.;
    // a[1] -= fastCIC(dg, N, iX, iY - 1, iZ, dx, dy, dz, tx, ty, tz) * 8.;
    // a[1] += fastCIC(dg, N, iX, iY - 2, iZ, dx, dy, dz, tx, ty, tz);
    // a[1] *= fac_over_12;
    //
    // a[2] = 0.0;
    // a[2] -= fastCIC(dg, N, iX, iY, iZ + 2, dx, dy, dz, tx, ty, tz);
    // a[2] += fastCIC(dg, N, iX, iY, iZ + 1, dx, dy, dz, tx, ty, tz) * 8.;
    // a[2] -= fastCIC(dg, N, iX, iY, iZ - 1, dx, dy, dz, tx, ty, tz) * 8.;
    // a[2] += fastCIC(dg, N, iX, iY, iZ - 2, dx, dy, dz, tx, ty, tz);
    // a[2] *= fac_over_12;

}


/* Compute the acceleration from the potential grid using CIC interpolation */
/* Uses a second-order accurate central difference scheme */
void accelCIC_2nd(const struct distributed_grid *dg, double *x, double *a) {

    /* Physical length to grid conversion factor */
    int N = dg->N;
    double boxlen = dg->boxlen;
    double fac = N / boxlen;
    double fac_over_2 = 0.5 * fac;

    /* Coordinates are mapped to [0, N] */
    double X = x[0];
    double Y = x[1];
    double Z = x[2];

    /* Integer grid position (floor not needed, as wrapping is ensured) */
    int iX = X;
    int iY = Y;
    int iZ = Z;

    /* Displacements from grid corner */
    double dx = X - iX;
    double dy = Y - iY;
    double dz = Z - iZ;
    double tx = 1.0 - dx;
    double ty = 1.0 - dy;
    double tz = 1.0 - dz;

    /* Products of fractional displacements from cell corners */
    double ttt = tx * ty * tz;
    double ttd = tx * ty * dz;
    double dtt = dx * ty * tz;
    double dtd = dx * ty * dz;
    double tdt = tx * dy * tz;
    double tdd = tx * dy * dz;
    double ddt = dx * dy * tz;
    double ddd = dx * dy * dz;

    /* Wrap the integer coordinates (not necessary for x) */
    int iX0 = iX - 1 - dg->X0 + dg->buffer_width;
    int iY0 = wrap(iY - 1, N);
    int iZ0 = wrap(iZ - 1, N);
    int iX2 = iX + 1 - dg->X0 + dg->buffer_width;
    int iY2 = wrap(iY + 1, N);
    int iZ2 = wrap(iZ + 1, N);
    int iX3 = iX + 2 - dg->X0 + dg->buffer_width;
    int iY3 = wrap(iY + 2, N);
    int iZ3 = wrap(iZ + 2, N);

    iX += - dg->X0 + dg->buffer_width;

    GridFloatType *box = dg->buffered_box;
    int Nz = dg->Nz;

    /* Retrieve the values necessary for the finite difference scheme */
    double val_111 = box[row_major_index(iX, iY, iZ, N, Nz)];
    double val_121 = box[row_major_index(iX, iY2, iZ, N, Nz)];
    double val_223 = box[row_major_index(iX2, iY2, iZ3, N, Nz)];
    double val_110 = box[row_major_index(iX, iY, iZ0, N, Nz)];
    double val_022 = box[row_major_index(iX0, iY2, iZ2, N, Nz)];
    double val_221 = box[row_major_index(iX2, iY2, iZ, N, Nz)];
    double val_222 = box[row_major_index(iX2, iY2, iZ2, N, Nz)];
    double val_201 = box[row_major_index(iX2, iY0, iZ, N, Nz)];
    double val_123 = box[row_major_index(iX, iY2, iZ3, N, Nz)];
    double val_021 = box[row_major_index(iX0, iY2, iZ, N, Nz)];
    double val_131 = box[row_major_index(iX, iY3, iZ, N, Nz)];
    double val_120 = box[row_major_index(iX, iY2, iZ0, N, Nz)];
    double val_232 = box[row_major_index(iX2, iY3, iZ2, N, Nz)];
    double val_132 = box[row_major_index(iX, iY3, iZ2, N, Nz)];
    double val_312 = box[row_major_index(iX3, iY, iZ2, N, Nz)];
    double val_101 = box[row_major_index(iX, iY0, iZ, N, Nz)];
    double val_202 = box[row_major_index(iX2, iY0, iZ2, N, Nz)];
    double val_220 = box[row_major_index(iX2, iY2, iZ0, N, Nz)];
    double val_113 = box[row_major_index(iX, iY, iZ3, N, Nz)];
    double val_112 = box[row_major_index(iX, iY, iZ2, N, Nz)];
    double val_210 = box[row_major_index(iX2, iY, iZ0, N, Nz)];
    double val_122 = box[row_major_index(iX, iY2, iZ2, N, Nz)];
    double val_102 = box[row_major_index(iX, iY0, iZ2, N, Nz)];
    double val_213 = box[row_major_index(iX2, iY, iZ3, N, Nz)];
    double val_231 = box[row_major_index(iX2, iY3, iZ, N, Nz)];
    double val_311 = box[row_major_index(iX3, iY, iZ, N, Nz)];
    double val_211 = box[row_major_index(iX2, iY, iZ, N, Nz)];
    double val_011 = box[row_major_index(iX0, iY, iZ, N, Nz)];
    double val_012 = box[row_major_index(iX0, iY, iZ2, N, Nz)];
    double val_321 = box[row_major_index(iX3, iY2, iZ, N, Nz)];
    double val_212 = box[row_major_index(iX2, iY, iZ2, N, Nz)];
    double val_322 = box[row_major_index(iX3, iY2, iZ2, N, Nz)];

    /* Compute the finite difference along the x-axis */
    a[0] += (val_211 - val_011) * ttt;
    a[0] += (val_212 - val_012) * ttd;
    a[0] += (val_311 - val_111) * dtt;
    a[0] += (val_312 - val_112) * dtd;
    a[0] += (val_221 - val_021) * tdt;
    a[0] += (val_222 - val_022) * tdd;
    a[0] += (val_321 - val_121) * ddt;
    a[0] += (val_322 - val_122) * ddd;

    /* Compute the finite difference along the y-axis */
    a[1] += (val_121 - val_101) * ttt;
    a[1] += (val_122 - val_102) * ttd;
    a[1] += (val_221 - val_201) * dtt;
    a[1] += (val_222 - val_202) * dtd;
    a[1] += (val_131 - val_111) * tdt;
    a[1] += (val_132 - val_112) * tdd;
    a[1] += (val_231 - val_211) * ddt;
    a[1] += (val_232 - val_212) * ddd;

    /* Compute the finite difference along the z-axis */
    a[2] += (val_112 - val_110) * ttt;
    a[2] += (val_113 - val_111) * ttd;
    a[2] += (val_212 - val_210) * dtt;
    a[2] += (val_213 - val_211) * dtd;
    a[2] += (val_122 - val_120) * tdt;
    a[2] += (val_123 - val_121) * tdd;
    a[2] += (val_222 - val_220) * ddt;
    a[2] += (val_223 - val_221) * ddd;

    a[0] *= fac_over_2;
    a[1] *= fac_over_2;
    a[2] *= fac_over_2;


    // a[0] = 0.0;
    // a[0] += fastCIC(dg, N, iX + 1, iY, iZ, dx, dy, dz, tx, ty, tz);
    // a[0] -= fastCIC(dg, N, iX - 1, iY, iZ, dx, dy, dz, tx, ty, tz);
    // a[0] *= fac_over_2;
    //
    // a[1] = 0.0;
    // a[1] += fastCIC(dg, N, iX, iY + 1, iZ, dx, dy, dz, tx, ty, tz);
    // a[1] -= fastCIC(dg, N, iX, iY - 1, iZ, dx, dy, dz, tx, ty, tz);
    // a[1] *= fac_over_2;
    //
    // a[2] = 0.0;
    // a[2] += fastCIC(dg, N, iX, iY, iZ + 1, dx, dy, dz, tx, ty, tz);
    // a[2] -= fastCIC(dg, N, iX, iY, iZ - 1, dx, dy, dz, tx, ty, tz);
    // a[2] *= fac_over_2;

}
