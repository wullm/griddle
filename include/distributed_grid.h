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

#ifndef DISTRIBUTED_GRID_H
#define DISTRIBUTED_GRID_H

#include <mpi.h>
#include <fftw3-mpi.h>

#include "fft_types.h"

#define DEFAULT_BUFFER_WIDTH 3

#define wrap(i,N) ((i)%(N)+(N))%(N)
#define fwrap(x,L) fmod(fmod((x),(L))+(L),(L))

struct distributed_grid {

    /* Global attributes (equal on all MPI ranks) */
    long int N;
    double boxlen;
    MPI_Comm comm;
    char momentum_space; //track whether we are in momentum space

    long int Nx; // for the full real-space grid
    long int Ny; // for the full real-space grid
    long int Nz; // for the full real-space grid

    /* Local attributes */
    long int NX;
    long int X0;
    long int local_complex_size; //number of complex elements = NX * N * (N/2 + 1)
    long int local_real_size;
    long int local_real_size_with_buffers;

    /* Local portions of the complex and real arrays */
    GridComplexType *fbox;
    GridFloatType *box;
    GridFloatType *buffered_box;

    /* Additional buffers on the left and right (real only) */
    GridFloatType *buffer_left;
    GridFloatType *buffer_right;
    int buffer_width;

    /* GLOBAL SIZES:
     * fbox:    N * N * (N/2 + 1)               GridComplexType type
     * box:     N * N * (N + 2)                 GridFloatType type
     *
     * The real array is padded on the right in the Z-dimension.
     */

    /* LOCAL SIZES:
     * fbox:    NX * N * (N/2 + 1)              GridComplexType type
     * box:     NX * N * (N + 2)                GridFloatType type
     *
     * The global arrays are sliced along the X-dimension. The local slice
     * corresponds to X0 <= X < X0 + NX.
     */
};

int alloc_local_grid(struct distributed_grid *dg, int N, double boxlen, MPI_Comm comm);
int alloc_local_grid_with_buffers(struct distributed_grid *dg, int N, double boxlen, int buffer_width, MPI_Comm comm);
int free_local_grid(struct distributed_grid *dg);
int free_local_real_grid(struct distributed_grid *dg);
int free_local_complex_grid(struct distributed_grid *dg);

int alloc_local_buffers(struct distributed_grid *dg, int buffer_width);
int free_local_buffers(struct distributed_grid *dg);
int create_local_buffers(struct distributed_grid *dg);
int add_local_buffers(struct distributed_grid *dg);

static inline GridFloatType* point_row_major_dg(int i, int j, int k, const struct distributed_grid *dg) {
    /* Wrap global coordinates */
    i = wrap(i,dg->N);
    j = wrap(j,dg->N);
    k = wrap(k,dg->N);

    /* Map to local slice (no out of bounds handling) */
    i = i - dg->X0;
    return &dg->box[i*dg->Ny*dg->Nz + j*dg->Nz + k];
}

static inline GridFloatType* point_row_major_dg_buffered(int i, int j, int k, const struct distributed_grid *dg) {
    /* Wrap global coordinates (but not the x-coordinate) */
    j = wrap(j,dg->N);
    k = wrap(k,dg->N);

    /* Map to local slice (no out of bounds handling) */
    i = i - dg->X0 + dg->buffer_width;
    return &dg->buffered_box[i*dg->Ny*dg->Nz + j*dg->Nz + k];
}

static inline GridFloatType* point_row_major_dg_buffered_nobounds(int i, int j, int k, const struct distributed_grid *dg) {
    /* Map to local slice (no out of bounds handling) */
    i = i - dg->X0 + dg->buffer_width;
    return &dg->buffered_box[i*dg->Ny*dg->Nz + j*dg->Nz + k];
}

static inline GridComplexType* point_row_major_half_dg(int i, int j, int k, const struct distributed_grid *dg) {
    /* Wrap global coordinates */
    i = wrap(i,dg->N);
    j = wrap(j,dg->N);
    k = wrap(k,dg->N/2 + 1);

    /* Map to local slice (no out of bounds handling) */
    i = i - dg->X0;
    return &dg->fbox[i*(dg->N/2+1)*dg->N + j*(dg->N/2+1) + k];
}

static inline GridComplexType* point_row_major_half_dg_nobounds(int i, int j, int k, const struct distributed_grid *dg) {
    /* Map to local slice (no out of bounds handling) */
    i = i - dg->X0;
    return &dg->fbox[i*(dg->N/2+1)*dg->N + j*(dg->N/2+1) + k];
}

static inline GridComplexType* point_row_major_half_dg_transposed(int i, int j, int k, const struct distributed_grid *dg) {
    /* Map to local slice (no out of bounds handling) */
    j = j - dg->X0;
    return &dg->fbox[j*(dg->N/2+1)*dg->N + i*(dg->N/2+1) + k];
}

#endif
