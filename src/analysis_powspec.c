/*******************************************************************************
 * This file is part of Sedulus.
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

#include <complex.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <mpi.h>

#include <assert.h>
#include <sys/time.h>

#include "../include/analysis_powspec.h"
#include "../include/message.h"
#include "../include/fft.h"
#include "../include/fft_kernels.h"
#include "../include/strooklat.h"

void calc_cross_powerspec_dist(const struct distributed_grid *dgrid1,
                               const struct distributed_grid *dgrid2,
                               int Y0, int NY, int bins, double *k_in_bins,
                               double *power_in_bins, int *obs_in_bins) {

    const int N = dgrid1->N;
    const int Nz_half = N/2 + 1;
    const double boxlen = dgrid1->boxlen;
    const double dk = 2*M_PI/boxlen;
    const double max_k = sqrt(3)*dk*N/2;
    const double min_k = dk;

    const double log_max_k = log(max_k);
    const double log_min_k = log(min_k);

    /* Reset the bins */
    for (int i = 0; i < bins; i++) {
        k_in_bins[i] = 0;
        power_in_bins[i] = 0;
        obs_in_bins[i] = 0;
    }

    /* Calculate the power spectrum */
    double kx,ky,kz,k;
    for (int y = Y0; y < Y0 + NY; y++) {
        for (int x = 0; x < N; x++) {
            for (int z = 0; z <= N/2; z++) {
                /* Calculate the wavevector */
                fft_wavevector(x, y, z, N, dk, &kx, &ky, &kz, &k);

                if (k==0) continue; //skip the DC mode

                /* Compute the bin */
                const float u = (log(k) - log_min_k) / (log_max_k - log_min_k);
                const int bin = floor((bins - 1) * u);

                assert(bin >= 0 && bin < bins);

                /* Compute the power <X,Y> with X,Y complex */                
                GridComplexType z1  = dgrid1->fbox[row_major_half_transposed(x, y - Y0, z, N, Nz_half)];
                GridComplexType z2  = dgrid2->fbox[row_major_half_transposed(x, y - Y0, z, N, Nz_half)];
                double a1 = creal(z1), a2 = creal(z2);
                double b1 = cimag(z1), b2 = cimag(z2);
                double Power = a1*a2 + b1*b2;

                /* All except the z=0 and the z=N/2 planes count double */
                int multiplicity = (z==0 || z==N/2) ? 1 : 2;

                /* Add to the tables */
                k_in_bins[bin] += multiplicity * k;
                power_in_bins[bin] += multiplicity * Power;
                obs_in_bins[bin] += multiplicity;
            }
        }
    }

    /* Still need to divide after communicating to obtain averages... */
}

/* TODO, kick and drift particles to the right time */
int analysis_powspec(struct distributed_grid *dgrid, int output_num,
                     double a_scale_factor, FourierPlanType r2c,
                     const struct units *us, const struct physical_consts *pcs,
                     const struct cosmology *cosmo, struct params *pars) {

    /* Get the dimensions of the cluster */
    int rank, MPI_Rank_Count;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &MPI_Rank_Count);
    
    /* Timer */
    struct timepair run_timer;
    timer_start(rank, &run_timer);

    /* The number of power spectrum bins */
    const int bins = pars->PowerSpectrumBins;
    const long int N = dgrid->N;

    /* Accumulate the total mass in the grid */
    double mass_tot_local = 0.;
    for (long int i = 0; i < dgrid->local_real_size; i++) {
        mass_tot_local += dgrid->box[i];
    }

    /* Communicate the total mass across all ranks */
    double mass_tot_global;
    MPI_Allreduce(&mass_tot_local, &mass_tot_global, 1, MPI_DOUBLE,
                  MPI_SUM, MPI_COMM_WORLD);

    /* Turn the mass grid into an overdensity grid */
    double avg_mass = mass_tot_global / ((double) N * N * N);
    for (long int i = 0; i < dgrid->local_real_size; i++) {
        dgrid->box[i] = dgrid->box[i] / avg_mass - 1.;
    }

    /* Prepare memory for the power spectrum calculation */
    double *k_in_bins = calloc(bins, sizeof(double));
    double *power_in_bins = calloc(bins, sizeof(double));
    int *obs_in_bins = calloc(bins, sizeof(int));
    
    /* Execute the Fourier transform */
    fft_execute(r2c);

    timer_stop(rank, &run_timer, "FFT (1) took ");

    /* The local slice of the real array is NX * N * (2*(N/2 + 1)). The x-index
     * runs over [X0, X0 + NX]. After the FFT, the complex is array is transposed
     * and the local slice is NY * N * (N/2 + 1), with the y-index running over
     * [Y0, Y0 + NY]. Note that the complex array has half the size. */
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
    const double boxvol = boxlen * boxlen * boxlen;
    const double dk = 2 * M_PI / boxlen;
    const double grid_fac = boxlen / N;
    const double fft_factor = boxvol / ((double) N * N * N);
    GridComplexType *fbox = dgrid->fbox;

    /* Make a look-up table for the inverse CIC kernel */
    GridFloatType *sinc_tab = malloc(N * sizeof(GridFloatType));
    for (int x = 0; x < N; x++) {
        double kx = (x > N/2) ? (x - N) * dk : x * dk;
        sinc_tab[x] = 1.0 / sinc(0.5 * kx * grid_fac);
    }

    timer_stop(rank, &run_timer, "Creating look-up table took ");

    /* Apply the inverse Poisson kernel (note that x and y are now transposed) */
    GridFloatType cx, cy, cz, ctot;
    for (int y = Y0; y < Y0 + NY; y++) {
        cy = sinc_tab[y];

        for (int x = 0; x < N; x++) {
            cx = sinc_tab[x];

            for (int z = 0; z <= N / 2; z++) {
                cz = sinc_tab[z];

                ctot = cx * cy * cz;
                ctot = ctot * ctot;

                GridComplexType kern = fft_factor * ctot;
                fbox[row_major_half_transposed(x, y - Y0, z, N, Nz_half)] *= kern;
            }
        }
    }

    timer_stop(rank, &run_timer, "Inverse CIC kernel took ");

    /* Free the look-up table */
    free(sinc_tab);

    /* Compute the power spectrum for the distributed grid */
    calc_cross_powerspec_dist(dgrid, dgrid, Y0, NY, bins, k_in_bins,
                              power_in_bins, obs_in_bins);

    timer_stop(rank, &run_timer, "Calculating the power spectrum took ");

    /* Prepare memory for reducing the power spectrum data */
    double *all_power_in_bins = NULL;
    double *all_k_in_bins = NULL;
    int *all_obs_in_bins = NULL;
    if (rank == 0) {
        all_power_in_bins  = calloc(bins, sizeof(double));
        all_k_in_bins = calloc(bins, sizeof(double));
        all_obs_in_bins = calloc(bins, sizeof(int));
    }

    /* Reduce the power spectrum data */
    MPI_Reduce(power_in_bins, all_power_in_bins, bins,
               MPI_DOUBLE, MPI_SUM, /* root = */ 0, MPI_COMM_WORLD);
    free(power_in_bins);

    /* Reduce the wavenumber array */
    MPI_Reduce(k_in_bins, all_k_in_bins, bins, MPI_DOUBLE,
               MPI_SUM, /* root = */ 0, MPI_COMM_WORLD);
    free(k_in_bins);

    /* Reduce the observations array */
    MPI_Reduce(obs_in_bins, all_obs_in_bins, bins, MPI_INT,
               MPI_SUM, /* root = */ 0, MPI_COMM_WORLD);
    free(obs_in_bins);

    if (rank == 0) {
        /* Divide to obtain averages */
        for (int i=0; i<bins; i++) {
            all_k_in_bins[i] /= all_obs_in_bins[i];
            all_power_in_bins[i] /= all_obs_in_bins[i];
            all_power_in_bins[i] /= boxvol;
        }
        
        
        /* First, let's clean up the data by removing empty bins */
        int nonzero_bins = 0;
        for (int i = 0; i < bins; i++) {
            if (all_obs_in_bins[i] > 0) nonzero_bins++;
        }

        message(rank, "We have %d non-empty bins.\n", nonzero_bins);

        /* Create array with valid wavenumbers and power (from non-empty bins) */
        double *valid_k = malloc(nonzero_bins * sizeof(double));
        double *valid_power = malloc(nonzero_bins * sizeof(double));
        int *valid_obs = malloc(nonzero_bins * sizeof(int));
        int valid_bin = 0;
        for (int i = 0; i < bins; i++) {
            if (all_obs_in_bins[i] > 0) {
                valid_k[valid_bin] = all_k_in_bins[i];
                valid_obs[valid_bin] = all_obs_in_bins[i];
                valid_power[valid_bin] = all_power_in_bins[i];
                valid_bin++;
            }
        }

        /* Unit conversion factors */
        const double k_unit = 1.0 / us->UnitLengthMetres;
        const double P_unit = 1.0 / (k_unit * k_unit * k_unit);
        const double k_unit_Mpc = MPC_METRES * k_unit;
        const double P_unit_Mpc = 1.0 / (k_unit_Mpc * k_unit_Mpc * k_unit_Mpc);

        /* Create a file to write the power spectrum data */
        char fname[50];
        sprintf(fname, "power_%04d.txt", output_num);
        FILE *f = fopen(fname, "w");

        /* Write the response data */
        fprintf(f, "# a = %g, z = %g, N = %ld\n", a_scale_factor, 1. / a_scale_factor - 1., dgrid->N);
        fprintf(f, "# k in units of U_L^-1 = %g m^-1 = %g Mpc^-1\n", k_unit, k_unit_Mpc);
        fprintf(f, "# P in units of U_L^3 = %g m^3 = %g Mpc^3\n", P_unit, P_unit_Mpc);
        fprintf(f, "# k P obs\n");
        for (int j = 0; j < nonzero_bins; j++) {
            fprintf(f, "%g %g %d\n", valid_k[j], valid_power[j], valid_obs[j]);
        }

        /* Close the file */
        fclose(f);

        /* Free the cleaned up arrays */
        free(valid_k);
        free(valid_obs);
        free(valid_power);
        free(all_k_in_bins);
        free(all_obs_in_bins);
        free(all_power_in_bins);
    }

    return 0;
}
