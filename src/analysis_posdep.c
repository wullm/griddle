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
#include "../include/fft_kernels.h"
#include "../include/strooklat.h"

#define DEBUG_CHECKS

void copy_local_grid(float *grid, const struct distributed_grid *dgrid,
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
                grid[row_major_padded(x - x0, y - y0, z - z0, N_sub)] = *point_row_major_dg(x, y, z, dgrid);
            }
        }
    }    
}

void calc_cross_powerspec(int N, double boxlen, const fftwf_complex *box1,
                          const fftwf_complex *box2, int bins,
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
int analysis_posdep(struct distributed_grid *dgrid, int output_num,
                    double a_scale_factor, const struct units *us,
                    const struct physical_consts *pcs,
                    const struct cosmology *cosmo, struct params *pars) {

    /* Get the dimensions of the cluster */
    int rank, MPI_Rank_Count;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &MPI_Rank_Count);

    /* Grid parameters */
    const int N = dgrid->N;
    const double boxlen = dgrid->boxlen;

    /* The number of sub-grids or cells */
    const int N_cells = pars->PositionDependentSplits;
    const int N_sub = N / N_cells;
    const double sublen = boxlen / N_cells;

    if (N_cells * N_sub != N) {
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
    const int bins = pars->PowerSpectrumBins;

    /* Prepare memory for the power spectrum calculation */
    double *k_in_bins = calloc(bins, sizeof(double));
    double *power_in_bins = calloc(num_cells * bins, sizeof(double));
    int *obs_in_bins = calloc(bins, sizeof(int));

    /* Prepare array with the total mass in each sub-grid */
    double *sub_masses = calloc(num_cells, sizeof(double));

    /* Memory for local copies of sub-grids */
    long int real_size = N_sub * N_sub * 2 * (N_sub / 2 + 1);
    float *grid = malloc(real_size * sizeof(float));
    float *temp = malloc(real_size * sizeof(float));

    /* Memory for a complex grid (always use in place transforms) */
    fftwf_complex *fgrid = (fftwf_complex*) grid;

    /* Always use single precision */
    fftwf_plan r2c = fftwf_plan_dft_r2c_3d(N_sub, N_sub, N_sub, grid, fgrid, PREPARE_FLAG);

    /* Loop until all sub-grids have been dealt with */
    for (int i = 0; i < iterations; i++) {
        
        int last_cell = (i + 1) * MPI_Rank_Count < num_cells ? (i + 1) * MPI_Rank_Count : num_cells;

        /* Communicate all sub-grids */
        for (int cell = i * MPI_Rank_Count; cell < last_cell; cell++) {
            int home_rank = cell % MPI_Rank_Count;
            int cell_i, cell_j, cell_k;
            inverse_row_major(cell, &cell_i, &cell_j, &cell_k, N_cells);

            /* Check if this cell has overlap with the local slice */
            const int x0 = cell_i * N_sub;
            int no_overlap = ((x0 < dgrid->X0 && x0 + N_sub < dgrid->X0) ||
                (x0 >= dgrid->X0 + dgrid->NX && x0 + N_sub >= dgrid->X0 + dgrid->NX));
            int engaged = !no_overlap || (home_rank == rank);

            /* Create a new communicator with only engaged ranks */
            int key = (home_rank == rank) ? 0 : 1; // home_rank -> rank 0 in the new comm
            MPI_Comm comm;
            MPI_Comm_split(MPI_COMM_WORLD, engaged, key, &comm);

            if (engaged) {
                /* Zero out the temporary grid */
                for (int l = 0; l < real_size; l++) {
                    temp[l] = 0;
                }

                /* Copy over the local grid */
                copy_local_grid(temp, dgrid, N_cells, N_sub, cell_i, cell_j, cell_k);

                /* Perform communications */
                MPI_Reduce(temp, grid, real_size, MPI_FLOAT, MPI_SUM, 0, comm);
            }

            /* Free the communicator */
            MPI_Comm_free(&comm);
        }

        /* Analyse the sub-grids */
        for (int cell = i * MPI_Rank_Count; cell < last_cell; cell++) {
            int home_rank = cell % MPI_Rank_Count;

            /* Only work on local grids */
            if (rank != home_rank) continue;

            /* Compute the total mass in the sub-grid */
            sub_masses[cell] = 0;
            for (int l = 0; l < real_size; l++) {
                sub_masses[cell] += grid[l];
            }

            /* Turn the mass grid into an over-density grid */
            double avg_mass = sub_masses[cell] / (N_sub * N_sub * N_sub);
            for (int l = 0; l < real_size; l++) {
                grid[l] = grid[l] / avg_mass - 1.0;
            }

            /* Fourier transform the sub-grid */
            fftwf_execute(r2c);
            fftf_normalize_r2c(fgrid, N_sub, sublen);

            /* Undo the CIC window function */
            struct Hermite_kern_params Hkp;
            Hkp.order = 2; //CIC
            Hkp.N = N_sub;
            Hkp.boxlen = sublen;

            /* Apply the inverse CIC kernel */
            fftf_apply_kernel(fgrid, fgrid, N_sub, sublen, kernel_undo_Hermite_window, &Hkp);

            /* Compute the power spectrum */
            calc_cross_powerspec(N_sub, sublen, fgrid, fgrid, bins, k_in_bins,
                                 power_in_bins + cell * bins, obs_in_bins);
        }
    }

    /* Clean up the FFT plan */
    fftwf_destroy_plan(r2c);

    /* Free the local grid */
    free(grid);
    free(temp);

    message(rank, "Done with position-dependent power spectra.\n");

    /* Prepare memory for reducing the power spectrum data */
    double *all_power_in_bins = NULL;
    double *all_sub_masses = NULL;
    if (rank == 0) {
        all_power_in_bins  = calloc(num_cells * bins, sizeof(double));
        all_sub_masses = calloc(num_cells, sizeof(double));
    }

    /* Reduce the power spectrum data */
    MPI_Reduce(power_in_bins, all_power_in_bins, num_cells * bins,
               MPI_DOUBLE, MPI_SUM, /* root = */ 0, MPI_COMM_WORLD);
    free(power_in_bins);

    /* Reduce the mass array */
    MPI_Reduce(sub_masses, all_sub_masses, num_cells, MPI_DOUBLE,
               MPI_SUM, /* root = */ 0, MPI_COMM_WORLD);
    free(sub_masses);

    if (rank == 0) {
        /* First, let's clean up the data by removing empty bins */
        int nonzero_bins = 0;
        for (int i = 0; i < bins; i++) {
            if (obs_in_bins[i] > 0) nonzero_bins++;
        }

        message(rank, "We have %d non-empty bins.\n", nonzero_bins);

        /* Create array with valid wavenumbers and power (from non-empty bins) */
        double *valid_k = malloc(nonzero_bins * sizeof(double));
        double *valid_power = malloc(nonzero_bins * num_cells * sizeof(double));
        int *valid_obs = malloc(nonzero_bins * sizeof(int));
        int valid_bin = 0;
        for (int i = 0; i < bins; i++) {
            if (obs_in_bins[i] > 0) {
                valid_k[valid_bin] = k_in_bins[i];
                valid_obs[valid_bin] = obs_in_bins[i];
                for (int cell = 0; cell < num_cells; cell++) {
                    valid_power[cell * nonzero_bins + valid_bin] = all_power_in_bins[cell * bins + i];
                }
                valid_bin++;
            }
        }

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
                isocymatic_power[i * nonzero_bins + j] = strooklat_interp(&spline_k, valid_power + (i * nonzero_bins), valid_k[j] * cbrt(1.0 + deltas[i]));
            }
        }

        /* Re-normalize the power, which scales as 1 / V ~ L^-3 ~ (1 + delta) */
        for (int i = 0; i < num_cells; i++) {
            for (int j = 0; j < nonzero_bins; j++) {
                isocymatic_power[i * nonzero_bins + j] *= 1.0 + deltas[i];
            }
        }

        /* Note: separate from the above length rescaling, we could renormalize
         * by another factor (1 + delta)^2, such that all spectra are relative
         * to the same global background density. We choose not to. */

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
        
        /* Unit conversion factors */
        const double m_unit = us->UnitMassKilogram;
        const double k_unit = 1.0 / us->UnitLengthMetres;
        const double P_unit = 1.0 / (k_unit * k_unit * k_unit);
        const double k_unit_Mpc = MPC_METRES * k_unit;
        const double P_unit_Mpc = 1.0 / (k_unit_Mpc * k_unit_Mpc * k_unit_Mpc);

        /* Create a file to write the response data */
        char fname[50];
        sprintf(fname, "response_%04d.txt", output_num);
        FILE *f = fopen(fname, "w");

        /* Write the response data */
        fprintf(f, "# a = %g, z = %g, N_cells = %d, N_sub = %d\n", a_scale_factor, 1. / a_scale_factor - 1., N_cells, N_sub);
        fprintf(f, "# k in units of U_L^-1 = %g m^-1 = %g Mpc^-1\n", k_unit, k_unit_Mpc);
        fprintf(f, "# P, Pi in units of U_L^3 = %g m^3 = %g Mpc^3\n", P_unit, P_unit_Mpc);
        fprintf(f, "# k B I <P> <Pi> <Pd> <Pid> obs\n");
        for (int j = 0; j < nonzero_bins; j++) {
            fprintf(f, "%g %g %g %g %g %g %g %d\n", valid_k[j], B[j], Bi[j], P[j], Pi[j], Pd[j], Pid[j], valid_obs[j]);
        }

        /* Close the file */
        fclose(f);

        /* Print all the raw spectra in one big table (bins as columns)*/
        char fname2[100];
        sprintf(fname2, "posdep_%04d.txt", output_num);
        f = fopen(fname2, "w");
        fprintf(f, "# Position-dependent power spectra for %d^3 sub-grids.\n", N_cells);
        fprintf(f, "# mass in units of U_M = %g kg\n", m_unit);
        fprintf(f, "# k in units of U_L^-1 = %g m^-1 = %g Mpc^-1\n", k_unit, k_unit_Mpc);
        fprintf(f, "# P in units of U_L^3 = %g m^3 = %g Mpc^3\n", P_unit, P_unit_Mpc);
        fprintf(f, "# grid mass P[k_0] P[k_1] ...\n");
        fprintf(f, "# k = NA NA ");
        for (int j = 0; j < nonzero_bins; j++) {
            fprintf(f, "%.8g ", valid_k[j]);
        }
        fprintf(f, "\n");
        fprintf(f, "# obs = NA NA ");
        for (int j = 0; j < nonzero_bins; j++) {
            fprintf(f, "%d ", valid_obs[j]);
        }
        fprintf(f, "\n");
        for (int i = 0; i < num_cells; i++) {
            fprintf(f, "%d ", i);
            fprintf(f, "%.8g ", all_sub_masses[i]);
            for (int j = 0; j < nonzero_bins; j++) {
                fprintf(f, "%.8g ", valid_power[i * nonzero_bins + j]);
            }
            fprintf(f, "\n");
        }
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
        free(valid_power);
        free(valid_k);
        free(valid_obs);
        free(all_sub_masses);
    }

    /* Free the cleaned up arrays */
    free(k_in_bins);
    free(obs_in_bins);

    return 0;
}
