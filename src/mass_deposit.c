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
#include "../include/neutrino.h"

const char *grid_type_names[num_grid_types] = {"all", "cb", "nu"};

int mass_deposition(struct distributed_grid *dgrid, struct particle *parts,
                    long long int local_partnum, enum grid_type gtype,
                    const struct cosmology *cosmo, const struct units *us,
                    const struct physical_consts *pcs, long int N_cb,
                    long int N_nu) {

    const long int N = dgrid->N;
    const long int Nz = dgrid->Nz;
    const int X0 = dgrid->X0;
    const int buffer_width = dgrid->buffer_width;
    const double boxlen = dgrid->boxlen;
    const double cell_factor = N / boxlen;
    const double cell_factor_3 = cell_factor * cell_factor * cell_factor;

    /* Conversion factor for neutrino momenta */
    const double neutrino_qfac = pcs->ElectronVolt / (pcs->SpeedOfLight * cosmo->T_nu_0 * pcs->kBoltzmann);

#ifndef WITH_MASSES
    /* Cosmological constants */
    const double h = cosmo->h;
    const double H_0 = h * 100 * KM_METRES / MPC_METRES * us->UnitTimeSeconds;
    const double rho_crit = 3.0 * H_0 * H_0 / (8. * M_PI * pcs->GravityG);
    const double Omega_cb = cosmo->Omega_cdm + cosmo->Omega_b;
    const double part_mass_cb = rho_crit * Omega_cb * pow(boxlen / N_cb, 3);

    /* Pull down the present day neutrino density per species */
    const double base_part_mass_nu = (N_nu > 0) ? rho_crit * pow(boxlen / N_nu, 3) : 0.;
    const double *Omega_nu_0 = cosmo->Omega_nu_0;
#endif

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

        /* Only count the relevant particles for this grid type */
        if (gtype == cb_mass && !compare_particle_type(part, cdm_type, 1)) continue;
        if (gtype == nu_mass && !compare_particle_type(part, neutrino_type, 0)) continue;

        double X = part->x[0] * int_to_grid_fac;
        double Y = part->x[1] * int_to_grid_fac;
        double Z = part->x[2] * int_to_grid_fac;

#ifdef WITH_MASSES
        double M = cell_factor_3 * part->m;
#else
        double M;
        if (compare_particle_type(part, cdm_type, 1)) {
            M = cell_factor_3 * part_mass_cb;
        } else if (compare_particle_type(part, neutrino_type, 0)) {
            M = cell_factor_3 * base_part_mass_nu * Omega_nu_0[neutrino_species(part, cosmo)];
        }
#endif

        /* Neutrino delta-f weighting */
        if (compare_particle_type(part, neutrino_type, 0)) {
            double q, w;
            neutrino_weight(part->v, part, cosmo, neutrino_qfac, &q, &w);
            M *= w;
        }

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
    const long int N = dgrid->N;
    const long int Nz_half = N/2 + 1;

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
#ifdef USE_CLASSIC_POISSON_KERNEL
    const double overall_fac = gravity_factor * fft_factor;
#else
    const double overall_fac = 0.5 * grid_fac * grid_fac * gravity_factor * fft_factor;
#endif
    GridComplexType *fbox = dgrid->fbox;

    /* Make look-up tables for the inverse Poisson and CIC kernels */
    double *pois_tab = malloc(N * sizeof(double));
    double *sinc_tab = malloc(N * sizeof(double));
    for (int x = 0; x < N; x++) {
        double kx = (x > N/2) ? (x - N) * dk : x * dk;

#ifdef USE_CLASSIC_POISSON_KERNEL
        pois_tab[x] = - kx * kx;
#else
        pois_tab[x] = cos(kx * grid_fac);
#endif

        double inv_sinc = 1.0 / sinc(0.5 * kx * grid_fac);
        double inv_sinc2 = inv_sinc * inv_sinc;
        sinc_tab[x] = inv_sinc2;
    }

    timer_stop(rank, &run_timer, "Creating look-up tables took ");

    /* Apply the inverse Poisson and CIC kernels (note that x and y are now
     * transposed) */
    double px, py, pz, ptot;
    double sx, sy, sz, stot;
    for (int y = Y0; y < Y0 + NY; y++) {
        py = pois_tab[y]; // Poisson
        sy = sinc_tab[y]; // CIC

        for (int x = 0; x < N; x++) {
            px = pois_tab[x]; // Poisson
            sx = sinc_tab[x]; // CIC

            for (int z = 0; z <= N / 2; z++) {
                pz = pois_tab[z]; // Poisson
                sz = sinc_tab[z]; // CIC

                ptot = px + py + pz; // Poisson
                stot = sx * sy * sz; // CIC

                double kern = 0.;
#ifdef USE_CLASSIC_POISSON_KERNEL
                if (ptot != 0.0) kern = overall_fac / ptot * stot;
#else
                if (ptot != 3.0) kern = overall_fac / (ptot - 3.0) * stot;
#endif

                fbox[row_major_half_transposed(x, y - Y0, z, N, Nz_half)] *= kern;
            }
        }
    }

    timer_stop(rank, &run_timer, "Inverse poisson kernel took ");

    /* Free the look-up tables */
    free(pois_tab);
    free(sinc_tab);

    /* Carry out the backward Fourier transform */
    // fft_c2r_dg(dgrid);

    /* Execute the Fourier transform */
    fft_execute(c2r);

    timer_stop(rank, &run_timer, "FFT (2) took ");

    return 0;
}
