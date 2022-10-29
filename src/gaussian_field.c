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

/* Methods for generating Gaussian random fields */

#include "../include/gaussian_field.h"
#include "../include/fft.h"

#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include <gsl/gsl_rng.h>


/* Generate a complex Gaussian random field */
int generate_complex_grf(struct distributed_grid *dg, rng_state *state) {
    /* The complex array is N * N * (N/2 + 1), locally we have NX * N * (N/2 + 1) */
    const int N = dg->N;
    const int NX = dg->NX; //the local slice is NX rows wide
    const int X0 = dg->X0; //the local slice starts at X = X0
    const double boxlen = dg->boxlen;
    const double boxvol = boxlen*boxlen*boxlen;
    const double factor = sqrt(boxvol/2);
    const double dk = 2 * M_PI / boxlen;

    /* Refer to fourier.pdf for details. */

    /* Because the Gaussian field is real, the Fourier transform fbox
     * is Hermitian. This can be stored with just N*N*(N/2+1) complex
     * numbers. The grid is divided over the nodes along the X-axis.
     * On this node, we have the slice X0 <= X < X0 + NX. Hence,
     * we loop over x in {X0, ..., X0 + NX - 1}, y in {0, ..., N-1},
     * and z in {0, ..., N/2}.
     */

    double kx,ky,kz,k;
    for (int x=X0; x<X0 + NX; x++) {
        for (int y=0; y<N; y++) {
            for (int z=0; z<=N/2; z++) {
                /* Calculate the wavevector */
                fft_wavevector(x, y, z, N, dk, &kx, &ky, &kz, &k);

                /* Ignore the constant DC mode */
                if (k > 0) {
                    double a = sampleNorm(state) * factor;
                    double b = sampleNorm(state) * factor;
                    dg->fbox[row_major_half_dg(x,y,z,dg)] = a + b * I;
                } else {
                    dg->fbox[row_major_half_dg(x,y,z,dg)] = 0;
                }
            }
        }
    }

    /* Right now, the grid is in momentum space */
    dg->momentum_space = 1;

    return 0;
}

/* Generate a complex Gaussian random field in the style of ngenIC. This is
 * essentially the same as generate_complex_grf, but with a different RNG.
 * Note also that the seed is now specified as an integer. */
int generate_ngeniclike_grf(struct distributed_grid *dg, int Seed) {
    /* The complex array is N * N * (N/2 + 1), locally we have NX * N * (N/2 + 1) */
    const int N = dg->N;
    const int NX = dg->NX; //the local slice is NX rows wide
    const int X0 = dg->X0; //the local slice starts at X = X0
    const double boxlen = dg->boxlen;
    const double boxvol = boxlen*boxlen*boxlen;
    const double factor = sqrt(boxvol/2);
    const double ngenic_factor = sqrt(2); //additional factor for our conventions
    const double dk = 2 * M_PI / boxlen;

    /* Refer to fourier.pdf for details. */

    /* Because the Gaussian field is real, the Fourier transform fbox
     * is Hermitian. This can be stored with just N*N*(N/2+1) complex
     * numbers. The grid is divided over the nodes along the X-axis.
     * On this node, we have the slice X0 <= X < X0 + NX. Hence,
     * we loop over x in {X0, ..., X0 + NX - 1}, y in {0, ..., N-1},
     * and z in {0, ..., N/2}.
     */

    /* Allocate a table of RNG seeds */
    unsigned int *seedtable = malloc(N * N * sizeof(unsigned int));

    /* GSL random number generator */
    gsl_rng *random_generator = gsl_rng_alloc(gsl_rng_ranlxd1);
    gsl_rng_set(random_generator, Seed);

    /* Generate a table of RNG seeds like ngenIC */

    for (int i = 0; i < N / 2; i++) {
        for (int j = 0; j < i; j++)
            seedtable[i * N + j] = 0x7fffffff * gsl_rng_uniform(random_generator);
        for (int j = 0; j < i + 1; j++)
            seedtable[j * N + i] = 0x7fffffff * gsl_rng_uniform(random_generator);
        for (int j = 0; j < i; j++)
	        seedtable[(N - 1 - i) * N + j] = 0x7fffffff * gsl_rng_uniform(random_generator);
        for (int j = 0; j < i + 1; j++)
	        seedtable[(N - 1 - j) * N + i] = 0x7fffffff * gsl_rng_uniform(random_generator);
        for (int j = 0; j < i; j++)
	        seedtable[i * N + (N - 1 - j)] = 0x7fffffff * gsl_rng_uniform(random_generator);
        for (int j = 0; j < i + 1; j++)
	        seedtable[j * N + (N - 1 - i)] = 0x7fffffff * gsl_rng_uniform(random_generator);
        for (int j = 0; j < i; j++)
	        seedtable[(N - 1 - i) * N + (N - 1 - j)] = 0x7fffffff * gsl_rng_uniform(random_generator);
        for (int j = 0; j < i + 1; j++)
	        seedtable[(N - 1 - j) * N + (N - 1 - i)] = 0x7fffffff * gsl_rng_uniform(random_generator);
    }

    double kx,ky,kz,k;
    for (int x=X0; x<X0 + NX; x++) {
        for (int y=0; y<N; y++) {
            /* Update the RNG with the seed for this 2D slice */
            gsl_rng_set (random_generator, seedtable[x * N + y]);

            for (int z=0; z<=N/2; z++) {
                /* Calculate the wavevector */
                fft_wavevector(x, y, z, N, dk, &kx, &ky, &kz, &k);

                /* Sample the phase and amplitude */
                double phase = gsl_rng_uniform(random_generator) * 2 * M_PI;
                double ampl = gsl_rng_uniform(random_generator);
                assert(ampl != 0);

                /* Ignore the constant DC mode */
                if (k > 0) {
                    double delta = sqrt(-log(ampl)) * factor * ngenic_factor;
                    double a = delta * sin (phase);
                    double b = delta * cos (phase);
                    dg->fbox[row_major_half_dg(x,y,z,dg)] = a + b * I;
                } else {
                    dg->fbox[row_major_half_dg(x,y,z,dg)] = 0;
                }
            }
        }
    }

    /* Right now, the grid is in momentum space */
    dg->momentum_space = 1;

    return 0;
}

/* Perform corrections to the generated Gaussian random field such that the
 * complex array is truly Hermitian. This only affects the planes k_z = 0
 * and k_z = N/2.
 *
 * Because the grid is divided over several MPI ranks along the X-axis, we
 * first collect the k_z = 0 and k_z = N/2 planes in full on each node. Then,
 * we make the necessary corrections. */
int enforce_hermiticity(struct distributed_grid *dg) {
    /* The complex array is N * N * (N/2 + 1), locally we have NX * N * (N/2 + 1) */
    const int N = dg->N;
    const int NX = dg->NX; //the local slice is NX rows wide
    const int X0 = dg->X0; //the local slice starts at X = X0
    const int slice_size = NX * N;
    const int slice_offset = X0 * N;

    /* Get the number of ranks */
    int MPI_Rank_Count;
    MPI_Comm_size(MPI_COMM_WORLD, &MPI_Rank_Count);

    /* Get the X-dimension locations (X0's) of the slices on each rank */
    int *slice_sizes = malloc(MPI_Rank_Count * sizeof(int));
    int *slice_offsets = malloc(MPI_Rank_Count * sizeof(int));
    MPI_Allgather(&slice_size, 1, MPI_INT, slice_sizes, 1, MPI_INT, dg->comm);
    MPI_Allgather(&slice_offset, 1, MPI_INT, slice_offsets, 1, MPI_INT, dg->comm);

    /* The first (k=0) and last (k=N/2+1) planes need hermiticity enforced */

    /* Collect the plane on all nodes */
    GridComplexType *our_slice = fft_alloc_complex(NX * N);
    GridComplexType *full_plane = fft_alloc_complex(N * N);

    /* For both planes */
    for (int z=0; z<=N/2; z+=N/2) { //runs over z=0 and z=N/2

        /* Fill our local slice of the plane */
        for (int x=X0; x<X0 + NX; x++) {
            for (int y=0; y<N; y++) {
                long long int id = row_major_half_dg(x, y, z, dg);
                our_slice[(x-X0)*N + y] = dg->fbox[id];
            }
        }

        /* Gather all the slices on all the nodes */
        MPI_Allgatherv(our_slice, NX * N, MPI_COMPLEX_GRID_TYPE, full_plane,
                       slice_sizes, slice_offsets, MPI_COMPLEX_GRID_TYPE, dg->comm);

        /* Enforce hermiticity: f(k) = f*(-k) */
        for (int x=X0; x<X0 + NX; x++) {
            for (int y=0; y<N; y++) {
                if (x != 0 && x < N/2) continue; //skip the lower half
                if ((x == 0 || x == N/2) && y < N/2) continue; //skip two strips


                int invx = (x > 0) ? N - x : 0;
                int invy = (y > 0) ? N - y : 0;
                int invz = (z > 0) ? N - z : 0; //maps 0->0 and (N/2)->(N/2)

                long long int id = row_major_half_dg(x,y,z,dg);

                /* If the point maps to itself, throw away the imaginary part */
                if (invx == x && invy == y && invz == z) {
                    dg->fbox[id] = creal(dg->fbox[id]);
                } else {
                    /* Otherwise, set it to the conjugate of its mirror point */
                    dg->fbox[id] = conj(full_plane[invx*N + invy]);
                }
            }
        }

        /* Wait until all the ranks are finished */
        MPI_Barrier(dg->comm);
    }

    /* Free the memory */
    fft_free(our_slice);
    fft_free(full_plane);
    free(slice_sizes);
    free(slice_offsets);

    return 0;
}

int fix_and_pairing(struct distributed_grid *dg, char fixing, char inverting) {
    /* The complex array is N * N * (N/2 + 1), locally we have NX * N * (N/2 + 1) */
    const int N = dg->N;
    const int NX = dg->NX; //the local slice is NX rows wide
    const int X0 = dg->X0; //the local slice starts at X = X0
    const double sqrt2 = sqrt(2.0);
    const double boxlen = dg->boxlen;
    const double boxvol = boxlen*boxlen*boxlen;
    const double factor = sqrt(boxvol/2);

    /* Apply the fixing and/or inverting of Fourier modes */
    for (int x = X0; x < X0 + NX; x++) {
        for (int y = 0; y < N; y++) {
            for (int z = 0; z <= N/2; z++) {
                /* The current real and imaginary parts and absolute value */
                double a = creal(dg->fbox[row_major_half_dg(x,y,z,dg)]);
                double b = cimag(dg->fbox[row_major_half_dg(x,y,z,dg)]);
                double norm = hypot(a, b);

                /* Ignore the constant DC mode */
                if (norm > 0) {
                    double fixing_factor = (fixing ? factor * sqrt2 / norm : 1.0);
                    double invert_factor = (inverting ? -1.0 : 1.0);
                    double fact = fixing_factor * invert_factor;

                    dg->fbox[row_major_half_dg(x,y,z,dg)] = (a + b * I) * fact;
                }
            }
        }
    }

    return 0;
}
