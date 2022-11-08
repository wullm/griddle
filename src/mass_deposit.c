/*******************************************************************************
 * This file is part of Sedulus.
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

#include <complex.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "../include/mass_deposit.h"
#include "../include/fft.h"
#include "../include/fft_kernels.h"
#include "../include/message.h"

static inline long int row_major_index(int i, int j, int k, int N, int Nz) {
    return i*N*Nz + j*Nz + k;
}

static inline long int row_major_half_transposed(int i, int j, int k, int N, int Nz_half) {
    return j*Nz_half*N + i*Nz_half + k;
}

int mass_deposition_single(struct distributed_grid *dgrid,
                           struct particle *parts,
                           long long int local_partnum) {

    const long long int N = dgrid->N;
    const double boxlen = dgrid->boxlen;
    const double cell_factor = N / boxlen;
    const double cell_factor_3 = cell_factor * cell_factor * cell_factor;
    double total_mass = 0;

    /* Empty the grid */
    for (long int i = 0; i < dgrid->local_real_size; i++) {
        dgrid->box[i] = 0;
    }

    for (long long i = 0; i < local_partnum; i++) {
        struct particle *part = &parts[i];

        double X = part->x[0] / (boxlen/N);
        double Y = part->x[1] / (boxlen/N);
        double Z = part->x[2] / (boxlen/N);

#ifdef WITH_MASSES
        double M = part->m;
#else
        double M = 1.0;
#endif

        /* Neutrino delta-f weighting */
#ifdef WITH_PARTTYPE
        if (part->type == 6) {
            M *= part->w;
        }
#endif

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
                    int ii = iX + x;
                    int jj = iY + y;
                    int kk = iZ + z;

                    double xx = fabs(X - ii);
                    double yy = fabs(Y - jj);
                    double zz = fabs(Z - kk);

                    double part_x = xx < 1.0 ? (1.0 - xx) : 0.;
                    double part_y = yy < 1.0 ? (1.0 - yy) : 0.;
                    double part_z = zz < 1.0 ? (1.0 - zz) : 0.;

                    *point_row_major_dg(ii, jj, kk, dgrid) += M * cell_factor_3 * (part_x*part_y*part_z);
				}
			}
		}
    }

    return 0;
}

int mass_deposition(struct distributed_grid *dgrid, struct particle *parts,
                    long long int local_partnum) {

    const long long int N = dgrid->N;
    const int Nz = dgrid->Nz;
    const int X0 = dgrid->X0;
    const int buffer_width = dgrid->buffer_width;
    const double boxlen = dgrid->boxlen;
    const double cell_factor = N / boxlen;
    const double cell_factor_3 = cell_factor * cell_factor * cell_factor;

    /* Position factors */
    const double pos_to_int_fac = pow(2.0, POSITION_BITS) / boxlen;
    const double int_to_pos_fac = 1.0 / pos_to_int_fac;
    const double int_to_grid_fac = int_to_pos_fac * cell_factor;

    GridFloatType *box = dgrid->buffered_box;

    /* Empty the grid */
    for (long int i = 0; i < dgrid->local_real_size_with_buffers; i++) {
        dgrid->buffered_box[i] = 0.;
    }

    for (long long i = 0; i < local_partnum; i++) {
        struct particle *part = &parts[i];

        double X = part->x[0] * int_to_grid_fac;
        double Y = part->x[1] * int_to_grid_fac;
        double Z = part->x[2] * int_to_grid_fac;

#ifdef WITH_MASSES
        double M = part->m * cell_factor_3;
#else
        double M = cell_factor_3;
#endif

        /* Neutrino delta-f weighting */
#ifdef WITH_PARTTYPE
        if (part->type == 6) {
            M *= part->w;
        }
#endif

        /* The particles coordinates must be wrapped here! */
        int iX = X;
        int iY = Y;
        int iZ = Z;

#ifdef DEBUG_CHECKS
        if (iX < dgrid->X0 || iX >= dgrid->X0 + dgrid->NX) {
            printf("particle on the wrong rank %d %ld %ld\n", iX, dgrid->X0, dgrid->X0 + dgrid->NX);
        }
#endif

        /* Displacements from grid corner */
        double dx = X - iX;
        double dy = Y - iY;
        double dz = Z - iZ;
        double tx = 1.0 - dx;
        double ty = 1.0 - dy;
        double tz = 1.0 - dz;

        iX += - X0 + buffer_width;

        int iX2 = iX + 1;
        int iY2 = iY + 1;
        int iZ2 = iZ + 1;

        if (iY2 >= N) iY2 -= N;
        if (iZ2 >= N) iZ2 -= N;

        /* Deposit the mass over the nearest 8 cells */
        box[row_major_index(iX, iY, iZ, N, Nz)] += M * tx * ty * tz;
        box[row_major_index(iX, iY2, iZ, N, Nz)] += M * tx * dy * tz;
        box[row_major_index(iX, iY, iZ2, N, Nz)] += M * tx * ty * dz;
        box[row_major_index(iX, iY2, iZ2, N, Nz)] += M * tx * dy * dz;
        box[row_major_index(iX2, iY, iZ, N, Nz)] += M * dx * ty * tz;
        box[row_major_index(iX2, iY2, iZ, N, Nz)] += M * dx * dy * tz;
        box[row_major_index(iX2, iY, iZ2, N, Nz)] += M * dx * ty * dz;
        box[row_major_index(iX2, iY2, iZ2, N, Nz)] += M * dx * dy * dz;
    }

    return 0;
}

int compute_potential(struct distributed_grid *dgrid,
                      struct physical_consts *pcs, FourierPlanType r2c,
                      FourierPlanType c2r) {

    /* Get the dimensions of the cluster */
    int rank, MPI_Rank_Count;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &MPI_Rank_Count);

    /* Timer */
    struct timepair run_timer;
    timer_start(rank, &run_timer);

    /* Carry out the forward Fourier transform */
    // fft_r2c_dg(dgrid);

    /* Execute the Fourier transform */
    fft_execute(r2c);

    timer_stop(rank, &run_timer, "FFT (1) took ");

    /* The local slice of the real array is NX * N * (2*(N/2 + 1)). The x-index
     * runs over [X0, X0 + NX]. After the FFT, the complex is array is transposed
     * and the local slice is NY * N * (N/2 + 1), with the y-index running over
     * [Y0, Y0 + NY]. Note that the complex array has half the size. */
    const int N = dgrid->N;
    const int Nz_half = N/2 + 1;

    /* Get the local portion of the transposed array */
    long int NX, X0, NY, Y0;
    fftwf_mpi_local_size_3d_transposed(N, N, N, MPI_COMM_WORLD, &NX, &X0, &NY, &Y0);
#ifdef DEBUG_CHECKS
    /* We expect that NX == NY and X0 == Y0, since Nx == Ny in this problem */
    assert(NX == dgrid->NX);
    assert(X0 == dgrid->X0);
    assert(NX == NY);
    assert(X0 == Y0);
#endif

    /* Pull out other grid constants */
    const double boxlen = dgrid->boxlen;
    const double dk = 2 * M_PI / boxlen;
    const double grid_fac = boxlen / N;
    const double gravity_factor = -4.0 * M_PI * pcs->GravityG;
    const double fft_factor = 1.0 / ((double) N * N * N);
    const double overall_fac = 0.5 * grid_fac * grid_fac * gravity_factor * fft_factor;
    GridComplexType *fbox = dgrid->fbox;

    /* Make a look-up table for the cosines */
    GridFloatType *cos_tab = malloc(N * sizeof(GridFloatType));
    for (int x = 0; x < N; x++) {
        double kx = (x > N/2) ? (x - N) * dk : x * dk;
        cos_tab[x] = cos(kx * grid_fac);
    }

    timer_stop(rank, &run_timer, "Creating look-up table took ");

    /* Apply the inverse Poisson kernel (note that x and y are now transposed) */
    GridFloatType cx, cy, cz, ctot;
    for (int y = Y0; y < Y0 + NY; y++) {
        cy = cos_tab[y];

        for (int x = 0; x < N; x++) {
            cx = cos_tab[x];

            for (int z = 0; z <= N / 2; z++) {
                cz = cos_tab[z];

                ctot = cx + cy + cz;

                if (ctot != 3.0) {
                    GridComplexType kern = overall_fac / (ctot - 3.0);
                    fbox[row_major_half_transposed(x, y - Y0, z, N, Nz_half)] *= kern;
                }
            }
        }
    }

    timer_stop(rank, &run_timer, "Inverse poisson kernel took ");

    /* Free the look-up table */
    free(cos_tab);

    /* Carry out the backward Fourier transform */
    // fft_c2r_dg(dgrid);

    /* Execute the Fourier transform */
    fft_execute(c2r);

    timer_stop(rank, &run_timer, "FFT (2) took ");

    return 0;
}
