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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <mpi.h>

#include <assert.h>
#include <sys/time.h>

#include "../include/analysis_posdep.h"
#include "../include/message.h"
#include "../include/fft.h"
#include "../include/strooklat.h"

#define DEBUG_CHECKS

void copy_local_grid(GridFloatType *grid, const struct distributed_grid *dgrid,
                     int N_cells, int N_sub, int cell_i, int cell_j, int cell_k) {

    /* The start of the requested cell inside the global grid */
    const int x0 = cell_i * N_sub;
    const int y0 = cell_j * N_sub;
    const int z0 = cell_k * N_sub;
    
    /* Exit immediately if this cell has no overlap with the local slice */
    if ((x0 < dgrid->X0 && x0 + N_sub < dgrid->X0) ||
        (x0 >= dgrid->X0 + dgrid->NX && x0 + N_sub >= dgrid->X0 + dgrid->NX)) {
        return;
    }
    
    /* Copy over the local portion */
    for (int x = x0; x < x0 + N_sub; x++) {
        /* Skip slices not present on this rank */
        if (x < dgrid->X0) continue;
        if (x >= dgrid->X0 + dgrid->NX) continue;
        
        for (int y = y0; y < y0 + N_sub; y++) {
            for (int z = z0; z < z0 + N_sub; z++) {
                grid[row_major(x - x0, y - y0, z - z0, N_sub)] = *point_row_major_dg(x, y, z, dgrid);
            }
        }
    }    
}

void calc_cross_powerspec(int N, double boxlen, const GridComplexType *box1,
                          const GridComplexType *box2, int bins,
                          double *k_in_bins, double *power_in_bins,
                          int *obs_in_bins) {

    const double boxvol = boxlen*boxlen*boxlen;
    const double dk = 2*M_PI/boxlen;
    const double max_k = sqrt(3)*dk*N/2;
    const double min_k = dk;

    const double log_max_k = log(max_k);
    const double log_min_k = log(min_k);

    /* Reset the bins */
    for (int i=0; i<bins; i++) {
        k_in_bins[i] = 0;
        power_in_bins[i] = 0;
        obs_in_bins[i] = 0;
    }

    /* Calculate the power spectrum */
    double kx,ky,kz,k;
    for (int x=0; x<N; x++) {
        for (int y=0; y<N; y++) {
            for (int z=0; z<=N/2; z++) {
                /* Calculate the wavevector */
                fft_wavevector(x, y, z, N, dk, &kx, &ky, &kz, &k);

                if (k==0) continue; //skip the DC mode

                /* Compute the bin */
                const float u = (log(k) - log_min_k) / (log_max_k - log_min_k);
                const int bin = floor((bins - 1) * u);
                const long long int id = row_major_half(x, y, z, N);

                assert(bin >= 0 && bin < bins);

                /* Compute the power <X,Y> with X,Y complex */
                double a1 = box1[id][0], a2 = box2[id][0];
                double b1 = box1[id][1], b2 = box2[id][1];
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

    /* Divide to obtain averages */
    for (int i=0; i<bins; i++) {
        k_in_bins[i] /= obs_in_bins[i];
        power_in_bins[i] /= obs_in_bins[i];
        power_in_bins[i] /= boxvol;
    }
}

/* TODO, kick and drift particles to the right time */
int analysis_posdep(struct distributed_grid *dgrid, double boxlen, 
                    long long int Ng, int output_num, double a_scale_factor,
                    const struct units *us, const struct physical_consts *pcs,
                    const struct cosmology *cosmo, struct params *pars) {

    /* Get the dimensions of the cluster */
    int rank, MPI_Rank_Count;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &MPI_Rank_Count);

    /* The number of sub-grids or cells */
    const int N_cells = 8;
    const int N_sub = Ng / N_cells;
    const double sublen = boxlen / N_cells;

    if (N_cells * N_sub != Ng) {
        printf("Error: The sub-grids do not divide into the main grid.\n");
        exit(1);
    }

    /* The total number of cells and iterations required */
    const int num_cells = N_cells * N_cells * N_cells;
    int iterations = num_cells / MPI_Rank_Count;
    if (iterations * MPI_Rank_Count != num_cells) {
        iterations++;
    }

    /* The number of power spectrum bins */
    const int bins = 50;

    /* Prepare memory for the power spectrum calculation */
    double *k_in_bins = calloc(bins, sizeof(double));
    double *power_in_bins = calloc(num_cells * bins, sizeof(double));
    int *obs_in_bins = calloc(bins, sizeof(int));

    /* Prepare array with the total mass in each sub-grid */
    double *sub_masses = calloc(num_cells, sizeof(double));

    /* Memory for local copies of sub-grids */
    GridFloatType *grid = malloc(N_sub * N_sub * N_sub * sizeof(GridFloatType));
    GridFloatType *temp = malloc(N_sub * N_sub * N_sub * sizeof(GridFloatType));
    bzero(grid, N_sub * N_sub * N_sub * sizeof(GridFloatType));
    bzero(temp, N_sub * N_sub * N_sub * sizeof(GridFloatType));

    /* Memory for a complex grid */
    GridComplexType *fgrid = malloc(N_sub * N_sub * (N_sub / 2 + 1) * sizeof(GridComplexType));

#ifdef SINGLE_PRECISION_FFTW
    fftwf_plan r2c = fftwf_plan_dft_r2c_3d(N_sub, N_sub, N_sub, grid, fgrid, FFTW_ESTIMATE);
#else
    fftw_plan r2c = fftw_plan_dft_r2c_3d(N_sub, N_sub, N_sub, grid, fgrid, FFTW_ESTIMATE);
#endif

    /* Loop until all sub-grids have been dealt with */
    for (int i = 0; i < iterations; i++) {
        
        int last_cell = (i + 1) * MPI_Rank_Count < num_cells ? (i + 1) * MPI_Rank_Count : num_cells;

        /* Communicate all sub-grids */
        for (int cell = i * MPI_Rank_Count; cell < last_cell; cell++) {
            int home_rank = cell % MPI_Rank_Count;
            int cell_i, cell_j, cell_k;
            inverse_row_major(cell, &cell_i, &cell_j, &cell_k, N_cells);

            /* Copy over the local grid */
            copy_local_grid(temp, dgrid, N_cells, N_sub, cell_i, cell_j, cell_k);

            /* Perform communications */
            MPI_Reduce(temp, grid, N_sub * N_sub * N_sub, MPI_GRID_TYPE,
                       MPI_SUM, home_rank, MPI_COMM_WORLD);
        }

        /* Analyse the sub-grids */
        for (int cell = i * MPI_Rank_Count; cell < last_cell; cell++) {
            int home_rank = cell % MPI_Rank_Count;

            /* Only work on local grids */
            if (rank != home_rank) continue;

            /* Compute the total mass in the sub-grid */
            sub_masses[cell] = 0;
            for (int l = 0; l < N_sub * N_sub * N_sub; l++) {
                sub_masses[cell] += grid[l];
            }

            /* Fourier transform the sub-grid */
            fft_execute(r2c);
            fft_normalize_r2c(fgrid, N_sub, sublen);

            /* Compute the power spectrum */
            calc_cross_powerspec(N_sub, sublen, fgrid, fgrid, bins, k_in_bins,
                                 power_in_bins + cell * bins, obs_in_bins);
        }
    }

    /* Clean up the FFT plan */
#ifdef SINGLE_PRECISION_FFTW
    fftwf_destroy_plan(r2c);
#else
    fftw_destroy_plan(r2c);
#endif

    message(rank, "Done with position-dependent power spectra.\n");

    /* First, let's clean up the data by removing empty bins */
    int nonzero_bins = 0;
    for (int i = 0; i < bins; i++) {
        if (obs_in_bins[i] > 0) nonzero_bins++;
    }

    message(rank, "We have %d non-empty bins.\n", nonzero_bins);

    /* Create array with valid wavenumbers and power (from non-empty bins) */
    double *valid_k = malloc(nonzero_bins * sizeof(double));
    double *valid_power = malloc(nonzero_bins * num_cells * sizeof(double));
    int valid_bin = 0;
    for (int i = 0; i < bins; i++) {
        if (obs_in_bins[i] > 0) {
            valid_k[valid_bin] = k_in_bins[i];
            for (int cell = 0; cell < num_cells; cell++) {
                valid_power[cell * nonzero_bins + valid_bin] = power_in_bins[cell * bins + i];
            }
            valid_bin++;
        }
    }

    /* Free the old power spectrum arrays */
    free(k_in_bins);
    free(power_in_bins);
    free(obs_in_bins);

    /* Prepare memory for reducing the power spectrum data */
    double *all_power_in_bins = NULL;
    double *all_sub_masses = NULL;
    if (rank == 0) {
        all_power_in_bins  = calloc(num_cells * nonzero_bins, sizeof(double));
        all_sub_masses = calloc(num_cells, sizeof(double));
    }

    /* Reduce the power spectrum data */
    MPI_Reduce(valid_power, all_power_in_bins, num_cells * nonzero_bins,
               MPI_DOUBLE, MPI_SUM, /* root = */ 0, MPI_COMM_WORLD);

    /* Reduce the mass array */
    MPI_Reduce(sub_masses, all_sub_masses, num_cells, MPI_DOUBLE,
               MPI_SUM, /* root = */ 0, MPI_COMM_WORLD);
    free(sub_masses);

    if (rank == 0) {
        /* Compute the total mass across all grids */
        double total_mass = 0.0;
        for (int i = 0; i < num_cells; i++) {
            total_mass += all_sub_masses[i];
        }

        /* Compute the large-scale density perturbations */
        double average_mass = total_mass / num_cells; 
        double *deltas = malloc(num_cells * sizeof(double));
        for (int i = 0; i < num_cells; i++) {
            deltas[i] = all_sub_masses[i] / average_mass - 1;
        }

        /* Prepare an interpolation spline for the wavenumbers */
        struct strooklat spline_k = {valid_k, nonzero_bins};
        init_strooklat_spline(&spline_k, 100);

        /* Compute the isocymatic power by interpolating */
        double *isocymatic_power  = calloc(num_cells * nonzero_bins, sizeof(double));
        for (int i = 0; i < num_cells; i++) {
            for (int j = 0; j < nonzero_bins; j++) {
                isocymatic_power[i * nonzero_bins + j] = strooklat_interp(&spline_k, valid_power + (i * nonzero_bins), valid_k[j] / cbrt(1.0 + deltas[i]));
            }
        }

        /* Free the spline */
        free_strooklat_spline(&spline_k);

        /* Compute the mean (isocymatic and ordinary) power */
        double *Pd = calloc(nonzero_bins, sizeof(double));
        double *P = calloc(nonzero_bins, sizeof(double));
        double *Pid = calloc(nonzero_bins, sizeof(double));
        double *Pi = calloc(nonzero_bins, sizeof(double));
        double dd = 0.;
        for (int i = 0; i < num_cells; i++) {
            for (int j = 0; j < nonzero_bins; j++) {
                Pd[j] += valid_power[i * nonzero_bins + j] * deltas[i] / num_cells;
                P[j] += valid_power[i * nonzero_bins + j] / num_cells;
                Pid[j] += isocymatic_power[i * nonzero_bins + j] * deltas[i] / num_cells;
                Pi[j] += isocymatic_power[i * nonzero_bins + j] / num_cells;
            }
            dd += deltas[i] * deltas[i] / num_cells;
        }

        /* Finally, compute the response functions */
        double *B = calloc(nonzero_bins, sizeof(double));
        double *Bi = calloc(nonzero_bins, sizeof(double));
        for (int j = 0; j < nonzero_bins; j++) {
            B[j] = Pd[j] / (P[j] * dd);
            Bi[j] = Pid[j] / (Pi[j] * dd);
        }
        
        /* Create a file to write the response data */
        char fname[50];
        sprintf(fname, "response_%04d_%03d.txt", output_num, rank);
        FILE *f = fopen(fname, "w");

        /* Write the response data */
        fprintf(f, "# a = %g, z = %g, N_cells = %d, N_sub = %d\n", a_scale_factor, 1. / a_scale_factor - 1., N_cells, N_sub);
        fprintf(f, "# k B I <P> <Pi> <Pd> <Pid>\n");
        for (int j = 0; j < nonzero_bins; j++) {
            fprintf(f, "%g %g %g %g %g %g %g\n", valid_k[j], B[j], Bi[j], P[j], Pi[j], Pd[j], Pid[j]);
        }

        /* Close the file */
        fclose(f);

        /* Free memory */
        free(Pd);
        free(P);
        free(Pid);
        free(Pi);
        free(B);
        free(Bi);
        free(deltas);
        free(isocymatic_power);
        free(all_power_in_bins);
        free(all_sub_masses);
    }

    /* Free the cleaned up arrays */
    free(valid_k);
    free(valid_power);

    /* Free the local grid */
    free(grid);
    free(temp);
    free(fgrid);

    return 0;
}
