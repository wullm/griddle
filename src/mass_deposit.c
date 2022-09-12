/*******************************************************************************
 * This file is part of Nyver.
 * Copyright (c) 2020 Willem Elbers (whe@willemelbers.com)
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
#include <string.h>
#include <assert.h>
#include <math.h>
#include "../include/mass_deposit.h"
#include "../include/fft.h"
#include "../include/fft_kernels.h"

int mass_deposition(struct distributed_grid *dgrid, struct particle *parts) {

    const long long int N = dgrid->N;
    const double boxlen = dgrid->boxlen;
    const double cell_factor = N / boxlen;
    const double cell_factor_3 = cell_factor * cell_factor * cell_factor;
    double total_mass = 0;

    /* Empty the grid */
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                dgrid->box[row_major_dg2(i, j, k, dgrid)] = 0;
            }
        }
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                struct particle *part = &parts[i * N * N + j * N + k];
                part->id = (long long int) i * N * N + j * N + k;

                double X = part->x[0] / (boxlen/N);
                double Y = part->x[1] / (boxlen/N);
                double Z = part->x[2] / (boxlen/N);

                double M = part->m;
                total_mass += M;

                int iX = (int) floor(X);
                int iY = (int) floor(Y);
                int iZ = (int) floor(Z);

                /* The search window with respect to the top-left-upper corner */
        		int lookLftX = (int) floor((X-iX) - 1);
        		int lookRgtX = (int) floor((X-iX) + 1);
        		int lookLftY = (int) floor((Y-iY) - 1);
        		int lookRgtY = (int) floor((Y-iY) + 1);
        		int lookLftZ = (int) floor((Z-iZ) - 1);
        		int lookRgtZ = (int) floor((Z-iZ) + 1);

                /* Do the mass assignment */
        		for (int x=lookLftX; x<=lookRgtX; x++) {
        			for (int y=lookLftY; y<=lookRgtY; y++) {
        				for (int z=lookLftZ; z<=lookRgtZ; z++) {
                            double xx = fabs(X - (iX+x));
                            double yy = fabs(Y - (iY+y));
                            double zz = fabs(Z - (iZ+z));

                            double part_x = xx < 1.0 ? (1.0 - xx) : 0.;
            				double part_y = yy < 1.0 ? (1.0 - yy) : 0.;
            				double part_z = zz < 1.0 ? (1.0 - zz) : 0.;

                            dgrid->box[row_major_dg2(iX+x, iY+y, iZ+z, dgrid)] += M * cell_factor_3 * (part_x*part_y*part_z);
        				}
        			}
        		}
            }
        }
    }

    // printf("The total mass is %g\n", total_mass);

    return 0;
}

int compute_potential(struct distributed_grid *dgrid,
                      struct physical_consts *pcs) {

    /* Carry out the forward Fourier transform */
    fft_r2c_dg(dgrid);

    /* Apply the inverse Poisson kernel */
    fft_apply_kernel_dg(dgrid, dgrid, kernel_inv_poisson, NULL);

    /* Multiply by Newton's constant */
    double boxlen = dgrid->boxlen;
    double factor = -4.0 * M_PI * pcs->GravityG;
    fft_apply_kernel_dg(dgrid, dgrid, kernel_constant, &factor);

    /* Carry out the backward Fourier transform */
    fft_c2r_dg(dgrid);

    return 0;
}
