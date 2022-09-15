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
#include <string.h>
#include <assert.h>
#include <math.h>
#include "../include/initial_conditions.h"
#include "../include/gaussian_field.h"
#include "../include/fft.h"
#include "../include/fft_kernels.h"
#include "../include/grid_io.h"
#include "../include/particle.h"
#include "../include/mesh_grav.h"

int generate_potential_grid(struct distributed_grid *dgrid, rng_state *seed,
                            char fix_modes, char invert_modes,
                            struct perturb_data *ptdat,
                            struct cosmology *cosmo, double z_start) {

    /* Find the transfer function index for CDM */
    int index_cdm = findTitle(ptdat->titles, "d_cdm", ptdat->n_functions);

    /* Generate a complex Hermitian Gaussian random field */
    generate_complex_grf(dgrid, seed);
    enforce_hermiticity(dgrid);

    /* Apply fixing and/or inverting of the modes for variance reduction? */
    if (fix_modes || invert_modes) {
        fix_and_pairing(dgrid, fix_modes, invert_modes);
    }

    /* Apply the primordial power spectrum without transfer functions */
    fft_apply_kernel_dg(dgrid, dgrid, kernel_power_no_transfer, cosmo);

    /* Pointer to the CDM density transfer function */
    double *tf = ptdat->delta + (ptdat->tau_size * ptdat->k_size) * index_cdm;

    /* Create interpolation splines for redshifts and wavenumbers */
    struct strooklat spline_z = {ptdat->redshift, ptdat->tau_size};
    struct strooklat spline_k = {ptdat->k, ptdat->k_size};
    init_strooklat_spline(&spline_z, 100);
    init_strooklat_spline(&spline_k, 100);

    /* Apply the CDM density transfer function */
    struct spline_params sp = {&spline_z, &spline_k, /* z = */ z_start, tf};
    fft_apply_kernel_dg(dgrid, dgrid, kernel_transfer_function, &sp);

    /* Compute the potential by applying the inverse Poisosn kernel */
    fft_apply_kernel_dg(dgrid, dgrid, kernel_inv_poisson_alt, NULL);

    /* Clean up strooklat interpolation splines */
    free_strooklat_spline(&spline_z);
    free_strooklat_spline(&spline_k);

    /* Execute the Fourier transform and normalize */
    fft_c2r_dg(dgrid);

    /* Export the GRF */
    // writeFieldFile_dg(dgrid, "grid.hdf5");

    return 0;
}

/* Compute the 2LPT potential. This requires two additional working memory
   grids to be allocated. */
int generate_2lpt_grid(struct distributed_grid *dgrid,
                       struct distributed_grid *temp1,
                       struct distributed_grid *temp2,
                       struct distributed_grid *dgrid_2lpt,
                       struct perturb_data *ptdat,
                       struct cosmology *cosmo, double z_start) {

    /* Execute the Fourier transform and normalize */
    fft_r2c_dg(dgrid);

    /* The complex array is N * N * (N/2 + 1), locally we have NX * N * (N/2 + 1) */
    const int N = dgrid->N;
    const int NX = dgrid->NX;
    const int X0 = dgrid->X0; //the local portion starts at X = X0

    /* We calculate derivatives using FFT kernels */
    const kernel_func derivatives[] = {kernel_dx, kernel_dy, kernel_dz};

    /* Erase the current data */
    for (int x = X0; x < X0 + NX; x++) {
        for (int y = 0; y < N; y++) {
            for (int z = 0; z < N; z++) {
                dgrid_2lpt->box[row_major_dg(x, y, z, dgrid)] = 0.;
            }
        }
    }

    /* First add the (xx)*(yy) + (xx)*(zz) + (yy)*(zz) terms */

    /* We need xy, xz, yz to compute the Hessian */
    const int index_a[] = {0, 0, 1};
    const int index_b[] = {1, 2, 2};

    /* Compute the 3 derivative components of the Hessian */
    for (int j=0; j<3; j++) {
        /* Compute the derivative d^2 phi / (dx_i dx_j) */
        fft_apply_kernel_dg(temp1, dgrid, derivatives[index_a[j]], NULL);
        fft_apply_kernel_dg(temp1, temp1, derivatives[index_a[j]], NULL);

        /* Compute the derivative d^2 phi / (dx_i dx_j) */
        fft_apply_kernel_dg(temp2, dgrid, derivatives[index_b[j]], NULL);
        fft_apply_kernel_dg(temp2, temp2, derivatives[index_b[j]], NULL);

        /* Fourier transform to configuration space */
        fft_c2r_dg(temp1);
        fft_c2r_dg(temp2);

        /* Add the product (=convolution) to the intermediate answer */
        for (int x = X0; x < X0 + NX; x++) {
            for (int y = 0; y < N; y++) {
                for (int z = 0; z < N; z++) {
                    long long int id = row_major_dg(x, y, z, dgrid);
                    dgrid_2lpt->box[id] += temp1->box[id] * temp1->box[id];
                }
            }
        }
    }

    /* Now add the (xy)*(xy) + (xz)*(xz) + (yz)*(yz) terms */

    /* Compute the 3 derivative components of the Hessian */
    for (int j=0; j<3; j++) {
        /* Compute the derivative d^2 phi / (dx_i dx_j) */
        fft_apply_kernel_dg(temp1, dgrid, derivatives[index_a[j]], NULL);
        fft_apply_kernel_dg(temp1, temp1, derivatives[index_b[j]], NULL);

        /* Fourier transform to configuration space */
        fft_c2r_dg(temp1);

        /* Subtract the square (=convolution) from the intermediate answer */
        for (int x = X0; x < X0 + NX; x++) {
            for (int y = 0; y < N; y++) {
                for (int z = 0; z < N; z++) {
                    long long int id = row_major_dg(x, y, z, dgrid);
                    dgrid_2lpt->box[id] -= temp1->box[id] * temp1->box[id];
                }
            }
        }
    }

    /* Fourier transform to momentum space */
    fft_r2c_dg(dgrid_2lpt);

    /* Compute the potential by applying the inverse Poisosn kernel */
    fft_apply_kernel_dg(dgrid_2lpt, dgrid_2lpt, kernel_inv_poisson_alt, NULL);

    /* Divide the potential by two */
    double factor = 0.5;
    fft_apply_kernel_dg(dgrid_2lpt, dgrid_2lpt, kernel_constant, &factor);

    /* Fourier transform to configuration space */
    fft_c2r_dg(dgrid_2lpt);
    fft_r2c_dg(dgrid);

    /* Export the 2LPT potential grid */
    // writeFieldFile_dg(dgrid_2lpt, "2lpt.hdf5");

    return 0;
}

int generate_particle_lattice(struct distributed_grid *lpt_potential,
                              struct distributed_grid *lpt_potential_2,
                              struct perturb_data *ptdat,
                              struct perturb_params *ptpars,
                              struct particle *parts, struct cosmology *cosmo,
                              struct units *us, struct physical_consts *pcs,
                              long long X0, long long NX, double z_start) {

    /* Create interpolation splines for redshifts and wavenumbers */
    struct strooklat spline_z = {ptdat->redshift, ptdat->tau_size};
    init_strooklat_spline(&spline_z, 100);

    /* The velocity factor a^2Hf (a^2 for the internal velocity variable) */
    const double a_start = 1.0 / (1.0 + z_start);
    const double f_start = strooklat_interp(&spline_z, ptdat->f_growth, z_start);
    const double H_start = strooklat_interp(&spline_z, ptdat->Hubble_H, z_start);
    const double vel_fact = a_start * a_start * f_start * H_start;

    /* The 2LPT factor */
    const double factor_2lpt = 3. / 7.;
    const double factor_vel_2lpt = factor_2lpt * 2.0;

    /* Grid constants */
    const int N = lpt_potential->N;
    const double boxlen = lpt_potential->boxlen;

    /* Cosmological constants */
    const double h = cosmo->h;
    const double H_0 = h * 100 * KM_METRES / MPC_METRES * us->UnitTimeSeconds;
    const double rho_crit = 3.0 * H_0 * H_0 / (8. * M_PI * pcs->GravityG);
    const double Omega_m = ptpars->Omega_m;
    const double part_mass = rho_crit * Omega_m * pow(boxlen / N, 3);

    /* Generate a particle lattice */
    for (int i = X0; i < X0+NX; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                struct particle *part = &parts[(i-X0) * N * N + j * N + k];
                part->id = (long long int) i * N * N + j * N + k;

                part->x[0] = i * boxlen / N;
                part->x[1] = j * boxlen / N;
                part->x[2] = k * boxlen / N;
                part->v[0] = 0.;
                part->v[1] = 0.;
                part->v[2] = 0.;

                /* Zel'dovich displacement */
                double dx[3] = {0,0,0};
                accelCIC(lpt_potential, N, boxlen, part->x, dx);

                /* The 2LPT displacement */
                double dx2[3] = {0,0,0};
                accelCIC(lpt_potential_2, N, boxlen, part->x, dx2);

                part->dx[0] = dx[0];
                part->dx[1] = dx[1];
                part->dx[2] = dx[2];
                part->dx2[0] = dx2[0];
                part->dx2[1] = dx2[1];
                part->dx2[2] = dx2[2];

                part->x[0] -= dx[0] + factor_2lpt * dx2[0];
                part->x[1] -= dx[1] + factor_2lpt * dx2[1];
                part->x[2] -= dx[2] + factor_2lpt * dx2[1];

                part->v[0] -= vel_fact * (dx[0] + factor_vel_2lpt * dx2[0]);
                part->v[1] -= vel_fact * (dx[1] + factor_vel_2lpt * dx2[1]);
                part->v[2] -= vel_fact * (dx[2] + factor_vel_2lpt * dx2[2]);
                part->m = part_mass;
            }
        }
    }

    /* Clean up strooklat interpolation splines */
    free_strooklat_spline(&spline_z);

    return 0;
}
