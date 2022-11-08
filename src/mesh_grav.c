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
/* Note: code generated with mesh_grav_generate.py. Do not change by hand */
void accelCIC(const struct distributed_grid *dg, double *x, double *a) {

    /* Physical length to grid conversion factor */
    int N = dg->N;
    double boxlen = dg->boxlen;
    double fac = N / boxlen;
    double fac_over_12 = fac / 12;

    GridFloatType *box = dg->buffered_box;
    int Nz = dg->Nz;

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

    /* Account for the distributed grid */
    iX += - dg->X0 + dg->buffer_width;

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
    int iY0 = iY - 2;
    int iZ0 = iZ - 2;
    if (iY0 < 0) iY0 -= N;
    if (iZ0 < 0) iZ0 -= N;
    int iX1 = iX - 1;
    int iY1 = iY - 1;
    int iZ1 = iZ - 1;
    if (iY1 < 0) iY1 -= N;
    if (iZ1 < 0) iZ1 -= N;
    int iX3 = iX + 1;
    int iY3 = iY + 1;
    int iZ3 = iZ + 1;
    if (iY3 >= N) iY3 -= N;
    if (iZ3 >= N) iZ3 -= N;
    int iX4 = iX + 2;
    int iY4 = iY + 2;
    int iZ4 = iZ + 2;
    if (iY4 >= N) iY4 -= N;
    if (iZ4 >= N) iZ4 -= N;
    int iX5 = iX + 3;
    int iY5 = iY + 3;
    int iZ5 = iZ + 3;
    if (iY5 >= N) iY5 -= N;
    if (iZ5 >= N) iZ5 -= N;

    /* Retrieve the values necessary for the finite difference scheme */
    double val_225 = box[row_major_index(iX, iY, iZ5, N, Nz)];
    double val_324 = box[row_major_index(iX3, iY, iZ4, N, Nz)];
    double val_023 = box[row_major_index(iX0, iY, iZ3, N, Nz)];
    double val_234 = box[row_major_index(iX, iY3, iZ4, N, Nz)];
    double val_523 = box[row_major_index(iX5, iY, iZ3, N, Nz)];
    double val_342 = box[row_major_index(iX3, iY4, iZ, N, Nz)];
    double val_212 = box[row_major_index(iX, iY1, iZ, N, Nz)];
    double val_123 = box[row_major_index(iX1, iY, iZ3, N, Nz)];
    double val_213 = box[row_major_index(iX, iY1, iZ3, N, Nz)];
    double val_352 = box[row_major_index(iX3, iY5, iZ, N, Nz)];
    double val_322 = box[row_major_index(iX3, iY, iZ, N, Nz)];
    double val_032 = box[row_major_index(iX0, iY3, iZ, N, Nz)];
    double val_325 = box[row_major_index(iX3, iY, iZ5, N, Nz)];
    double val_423 = box[row_major_index(iX4, iY, iZ3, N, Nz)];
    double val_422 = box[row_major_index(iX4, iY, iZ, N, Nz)];
    double val_432 = box[row_major_index(iX4, iY3, iZ, N, Nz)];
    double val_222 = box[row_major_index(iX, iY, iZ, N, Nz)];
    double val_122 = box[row_major_index(iX1, iY, iZ, N, Nz)];
    double val_335 = box[row_major_index(iX3, iY3, iZ5, N, Nz)];
    double val_532 = box[row_major_index(iX5, iY3, iZ, N, Nz)];
    double val_303 = box[row_major_index(iX3, iY0, iZ3, N, Nz)];
    double val_221 = box[row_major_index(iX, iY, iZ1, N, Nz)];
    double val_253 = box[row_major_index(iX, iY5, iZ3, N, Nz)];
    double val_333 = box[row_major_index(iX3, iY3, iZ3, N, Nz)];
    double val_334 = box[row_major_index(iX3, iY3, iZ4, N, Nz)];
    double val_302 = box[row_major_index(iX3, iY0, iZ, N, Nz)];
    double val_320 = box[row_major_index(iX3, iY, iZ0, N, Nz)];
    double val_353 = box[row_major_index(iX3, iY5, iZ3, N, Nz)];
    double val_033 = box[row_major_index(iX0, iY3, iZ3, N, Nz)];
    double val_332 = box[row_major_index(iX3, iY3, iZ, N, Nz)];
    double val_203 = box[row_major_index(iX, iY0, iZ3, N, Nz)];
    double val_022 = box[row_major_index(iX0, iY, iZ, N, Nz)];
    double val_433 = box[row_major_index(iX4, iY3, iZ3, N, Nz)];
    double val_252 = box[row_major_index(iX, iY5, iZ, N, Nz)];
    double val_533 = box[row_major_index(iX5, iY3, iZ3, N, Nz)];
    double val_312 = box[row_major_index(iX3, iY1, iZ, N, Nz)];
    double val_230 = box[row_major_index(iX, iY3, iZ0, N, Nz)];
    double val_132 = box[row_major_index(iX1, iY3, iZ, N, Nz)];
    double val_231 = box[row_major_index(iX, iY3, iZ1, N, Nz)];
    double val_343 = box[row_major_index(iX3, iY4, iZ3, N, Nz)];
    double val_202 = box[row_major_index(iX, iY0, iZ, N, Nz)];
    double val_235 = box[row_major_index(iX, iY3, iZ5, N, Nz)];
    double val_223 = box[row_major_index(iX, iY, iZ3, N, Nz)];
    double val_313 = box[row_major_index(iX3, iY1, iZ3, N, Nz)];
    double val_522 = box[row_major_index(iX5, iY, iZ, N, Nz)];
    double val_220 = box[row_major_index(iX, iY, iZ0, N, Nz)];
    double val_233 = box[row_major_index(iX, iY3, iZ3, N, Nz)];
    double val_232 = box[row_major_index(iX, iY3, iZ, N, Nz)];
    double val_242 = box[row_major_index(iX, iY4, iZ, N, Nz)];
    double val_243 = box[row_major_index(iX, iY4, iZ3, N, Nz)];
    double val_323 = box[row_major_index(iX3, iY, iZ3, N, Nz)];
    double val_330 = box[row_major_index(iX3, iY3, iZ0, N, Nz)];
    double val_331 = box[row_major_index(iX3, iY3, iZ1, N, Nz)];
    double val_224 = box[row_major_index(iX, iY, iZ4, N, Nz)];
    double val_133 = box[row_major_index(iX1, iY3, iZ3, N, Nz)];
    double val_321 = box[row_major_index(iX3, iY, iZ1, N, Nz)];

    /* Compute the finite difference along the x-axis */
    a[0] += (val_022 - val_422) * ttt;
    a[0] += (val_023 - val_423) * ttd;
    a[0] += (val_122 - val_522) * dtt;
    a[0] += (val_123 - val_523) * dtd;
    a[0] += (val_032 - val_432) * tdt;
    a[0] += (val_033 - val_433) * tdd;
    a[0] += (val_132 - val_532) * ddt;
    a[0] += (val_133 - val_533) * ddd;
    a[0] += (val_322 - val_122) * ttt * 8.0;
    a[0] += (val_323 - val_123) * ttd * 8.0;
    a[0] += (val_422 - val_222) * dtt * 8.0;
    a[0] += (val_423 - val_223) * dtd * 8.0;
    a[0] += (val_332 - val_132) * tdt * 8.0;
    a[0] += (val_333 - val_133) * tdd * 8.0;
    a[0] += (val_432 - val_232) * ddt * 8.0;
    a[0] += (val_433 - val_233) * ddd * 8.0;

    /* Compute the finite difference along the y-axis */
    a[1] += (val_202 - val_242) * ttt;
    a[1] += (val_203 - val_243) * ttd;
    a[1] += (val_302 - val_342) * dtt;
    a[1] += (val_303 - val_343) * dtd;
    a[1] += (val_212 - val_252) * tdt;
    a[1] += (val_213 - val_253) * tdd;
    a[1] += (val_312 - val_352) * ddt;
    a[1] += (val_313 - val_353) * ddd;
    a[1] += (val_232 - val_212) * ttt * 8.0;
    a[1] += (val_233 - val_213) * ttd * 8.0;
    a[1] += (val_332 - val_312) * dtt * 8.0;
    a[1] += (val_333 - val_313) * dtd * 8.0;
    a[1] += (val_242 - val_222) * tdt * 8.0;
    a[1] += (val_243 - val_223) * tdd * 8.0;
    a[1] += (val_342 - val_322) * ddt * 8.0;
    a[1] += (val_343 - val_323) * ddd * 8.0;

    /* Compute the finite difference along the z-axis */
    a[2] += (val_220 - val_224) * ttt;
    a[2] += (val_221 - val_225) * ttd;
    a[2] += (val_320 - val_324) * dtt;
    a[2] += (val_321 - val_325) * dtd;
    a[2] += (val_230 - val_234) * tdt;
    a[2] += (val_231 - val_235) * tdd;
    a[2] += (val_330 - val_334) * ddt;
    a[2] += (val_331 - val_335) * ddd;
    a[2] += (val_223 - val_221) * ttt * 8.0;
    a[2] += (val_224 - val_222) * ttd * 8.0;
    a[2] += (val_323 - val_321) * dtt * 8.0;
    a[2] += (val_324 - val_322) * dtd * 8.0;
    a[2] += (val_233 - val_231) * tdt * 8.0;
    a[2] += (val_234 - val_232) * tdd * 8.0;
    a[2] += (val_333 - val_331) * ddt * 8.0;
    a[2] += (val_334 - val_332) * ddd * 8.0;

    a[0] *= fac_over_12;
    a[1] *= fac_over_12;
    a[2] *= fac_over_12;
}


/* Compute the acceleration from the potential grid using CIC interpolation */
/* Uses a second-order accurate central difference scheme */
/* Note: code generated with mesh_grav_generate.py. Do not change by hand */
void accelCIC_2nd(const struct distributed_grid *dg, double *x, double *a) {

    /* Physical length to grid conversion factor */
    int N = dg->N;
    double boxlen = dg->boxlen;
    double fac = N / boxlen;
    double fac_over_2 = 0.5 * fac;

    GridFloatType *box = dg->buffered_box;
    int Nz = dg->Nz;

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

    /* Account for the distributed grid */
    iX += - dg->X0 + dg->buffer_width;

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
    int iX0 = iX - 1;
    int iY0 = iY - 1;
    int iZ0 = iZ - 1;
    if (iY0 < 0) iY0 -= N;
    if (iZ0 < 0) iZ0 -= N;
    int iX2 = iX + 1;
    int iY2 = iY + 1;
    int iZ2 = iZ + 1;
    if (iY2 >= N) iY2 -= N;
    if (iZ2 >= N) iZ2 -= N;
    int iX3 = iX + 2;
    int iY3 = iY + 2;
    int iZ3 = iZ + 2;
    if (iY3 >= N) iY3 -= N;
    if (iZ3 >= N) iZ3 -= N;

    /* Retrieve the values necessary for the finite difference scheme */
    double val_221 = box[row_major_index(iX2, iY2, iZ, N, Nz)];
    double val_312 = box[row_major_index(iX3, iY, iZ2, N, Nz)];
    double val_101 = box[row_major_index(iX, iY0, iZ, N, Nz)];
    double val_112 = box[row_major_index(iX, iY, iZ2, N, Nz)];
    double val_011 = box[row_major_index(iX0, iY, iZ, N, Nz)];
    double val_111 = box[row_major_index(iX, iY, iZ, N, Nz)];
    double val_322 = box[row_major_index(iX3, iY2, iZ2, N, Nz)];
    double val_021 = box[row_major_index(iX0, iY2, iZ, N, Nz)];
    double val_201 = box[row_major_index(iX2, iY0, iZ, N, Nz)];
    double val_210 = box[row_major_index(iX2, iY, iZ0, N, Nz)];
    double val_102 = box[row_major_index(iX, iY0, iZ2, N, Nz)];
    double val_110 = box[row_major_index(iX, iY, iZ0, N, Nz)];
    double val_211 = box[row_major_index(iX2, iY, iZ, N, Nz)];
    double val_131 = box[row_major_index(iX, iY3, iZ, N, Nz)];
    double val_223 = box[row_major_index(iX2, iY2, iZ3, N, Nz)];
    double val_123 = box[row_major_index(iX, iY2, iZ3, N, Nz)];
    double val_022 = box[row_major_index(iX0, iY2, iZ2, N, Nz)];
    double val_122 = box[row_major_index(iX, iY2, iZ2, N, Nz)];
    double val_321 = box[row_major_index(iX3, iY2, iZ, N, Nz)];
    double val_012 = box[row_major_index(iX0, iY, iZ2, N, Nz)];
    double val_311 = box[row_major_index(iX3, iY, iZ, N, Nz)];
    double val_220 = box[row_major_index(iX2, iY2, iZ0, N, Nz)];
    double val_113 = box[row_major_index(iX, iY, iZ3, N, Nz)];
    double val_132 = box[row_major_index(iX, iY3, iZ2, N, Nz)];
    double val_120 = box[row_major_index(iX, iY2, iZ0, N, Nz)];
    double val_222 = box[row_major_index(iX2, iY2, iZ2, N, Nz)];
    double val_212 = box[row_major_index(iX2, iY, iZ2, N, Nz)];
    double val_231 = box[row_major_index(iX2, iY3, iZ, N, Nz)];
    double val_232 = box[row_major_index(iX2, iY3, iZ2, N, Nz)];
    double val_121 = box[row_major_index(iX, iY2, iZ, N, Nz)];
    double val_202 = box[row_major_index(iX2, iY0, iZ2, N, Nz)];
    double val_213 = box[row_major_index(iX2, iY, iZ3, N, Nz)];

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
}

/* Compute the acceleration from the potential grid using CIC interpolation */
/* Uses a first-order accurate backward difference scheme */
/* Note: code generated with mesh_grav_generate.py. Do not change by hand */
void accelCIC_1st(const struct distributed_grid *dg, double *x, double *a) {

    /* Physical length to grid conversion factor */
    int N = dg->N;
    double boxlen = dg->boxlen;
    double fac = N / boxlen;

    GridFloatType *box = dg->buffered_box;
    int Nz = dg->Nz;

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
    
    /* Account for the distributed grid */
    iX += - dg->X0 + dg->buffer_width;

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
    int iX1 = iX - 1;
    int iY1 = iY - 1;
    int iZ1 = iZ - 1;
    if (iY1 < 0) iY1 -= N;
    if (iZ1 < 0) iZ1 -= N;
    int iX3 = iX + 1;
    int iY3 = iY + 1;
    int iZ3 = iZ + 1;
    if (iY3 >= N) iY3 -= N;
    if (iZ3 >= N) iZ3 -= N;

    /* Retrieve the values necessary for the finite difference scheme */
    double val_132 = box[row_major_index(iX1, iY3, iZ, N, Nz)];
    double val_233 = box[row_major_index(iX, iY3, iZ3, N, Nz)];
    double val_331 = box[row_major_index(iX3, iY3, iZ1, N, Nz)];
    double val_123 = box[row_major_index(iX1, iY, iZ3, N, Nz)];
    double val_222 = box[row_major_index(iX, iY, iZ, N, Nz)];
    double val_313 = box[row_major_index(iX3, iY1, iZ3, N, Nz)];
    double val_133 = box[row_major_index(iX1, iY3, iZ3, N, Nz)];
    double val_321 = box[row_major_index(iX3, iY, iZ1, N, Nz)];
    double val_223 = box[row_major_index(iX, iY, iZ3, N, Nz)];
    double val_322 = box[row_major_index(iX3, iY, iZ, N, Nz)];
    double val_221 = box[row_major_index(iX, iY, iZ1, N, Nz)];
    double val_333 = box[row_major_index(iX3, iY3, iZ3, N, Nz)];
    double val_213 = box[row_major_index(iX, iY1, iZ3, N, Nz)];
    double val_122 = box[row_major_index(iX1, iY, iZ, N, Nz)];
    double val_212 = box[row_major_index(iX, iY1, iZ, N, Nz)];
    double val_323 = box[row_major_index(iX3, iY, iZ3, N, Nz)];
    double val_231 = box[row_major_index(iX, iY3, iZ1, N, Nz)];
    double val_232 = box[row_major_index(iX, iY3, iZ, N, Nz)];
    double val_332 = box[row_major_index(iX3, iY3, iZ, N, Nz)];
    double val_312 = box[row_major_index(iX3, iY1, iZ, N, Nz)];

    /* Compute the finite difference along the x-axis */
    a[0] += (val_222 - val_122) * ttt;
    a[0] += (val_223 - val_123) * ttd;
    a[0] += (val_322 - val_222) * dtt;
    a[0] += (val_323 - val_223) * dtd;
    a[0] += (val_232 - val_132) * tdt;
    a[0] += (val_233 - val_133) * tdd;
    a[0] += (val_332 - val_232) * ddt;
    a[0] += (val_333 - val_233) * ddd;

    /* Compute the finite difference along the y-axis */
    a[1] += (val_222 - val_212) * ttt;
    a[1] += (val_223 - val_213) * ttd;
    a[1] += (val_322 - val_312) * dtt;
    a[1] += (val_323 - val_313) * dtd;
    a[1] += (val_232 - val_222) * tdt;
    a[1] += (val_233 - val_223) * tdd;
    a[1] += (val_332 - val_322) * ddt;
    a[1] += (val_333 - val_323) * ddd;

    /* Compute the finite difference along the z-axis */
    a[2] += (val_222 - val_221) * ttt;
    a[2] += (val_223 - val_222) * ttd;
    a[2] += (val_322 - val_321) * dtt;
    a[2] += (val_323 - val_322) * dtd;
    a[2] += (val_232 - val_231) * tdt;
    a[2] += (val_233 - val_232) * tdd;
    a[2] += (val_332 - val_331) * ddt;
    a[2] += (val_333 - val_332) * ddd;


    a[0] *= fac;
    a[1] *= fac;
    a[2] *= fac;
}
