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
#include "../include/distributed_grid.h"
#include "../include/fft.h"

int alloc_local_grid(struct distributed_grid *dg, int N, double boxlen, MPI_Comm comm) {
    /* Determine the size of the local portion */
    dg->local_complex_size = fft_mpi_local_size_3d(N, N, N/2+1, comm, &dg->NX, &dg->X0);
    dg->local_real_size = 2 * dg->local_complex_size;

    /* We do not have buffers */
    dg->local_real_size_with_buffers = dg->local_real_size;

    /* Store a reference to the communicator */
    dg->comm = comm;

    /* Store basic attributes */
    dg->boxlen = boxlen;
    dg->N = N;
    dg->Nx = N;
    dg->Ny = N;
    dg->Nz = 2*(N/2+1); // for the real grid

    /* Allocate memory for the real array */
    dg->box = fft_alloc_real(dg->local_real_size);

#ifdef USE_IN_PLACE_FFTS
    /* We will use in-place transforms */
    dg->fbox = (GridComplexType*) dg->box;
#else
    /* Allocate memory for the complex array */
    dg->fbox = fft_alloc_complex(dg->local_complex_size);
#endif

    /* This flag will be flipped each time we do a Fourier transform */
    dg->momentum_space = 0;

    /* By default, we do not allocate buffers */
    dg->buffer_width = 0;

    return 0;
}

int alloc_local_grid_with_buffers(struct distributed_grid *dg, int N, double boxlen, int buffer_width, MPI_Comm comm) {
    /* Determine the size of the local portion */
    dg->local_complex_size = fft_mpi_local_size_3d(N, N, N/2+1, comm, &dg->NX, &dg->X0);
    dg->local_real_size = 2 * dg->local_complex_size;

    /* Store basic attributes */
    dg->boxlen = boxlen;
    dg->N = N;
    dg->Nx = N;
    dg->Ny = N;
    dg->Nz = 2*(N/2+1); // for the real grid

    /* Also account for the buffers (2 buffers: left and right) */
    dg->local_real_size_with_buffers = dg->local_real_size + 2 * buffer_width * dg->Ny * dg->Nz;

    /* Store a reference to the communicator */
    dg->comm = comm;

    /* Allocate memory for the real array */
    dg->buffered_box = fft_alloc_real(dg->local_real_size_with_buffers);
    /* Point to where the local array actually begins (after the first buffer) */
    dg->box = dg->buffered_box + buffer_width * dg->Ny * dg->Nz;

#ifdef USE_IN_PLACE_FFTS
    /* We will use in-place transforms */
    dg->fbox = (GridComplexType*) dg->box;
#else
    /* Allocate memory for the complex array */
    dg->fbox = fft_alloc_complex(dg->local_complex_size);
#endif

    /* Pointers to the start of the left and right buffers */
    dg->buffer_left = dg->buffered_box;
    dg->buffer_right = dg->buffered_box + (dg->NX + buffer_width) * dg->Ny * dg->Nz;

    /* This flag will be flipped each time we do a Fourier transform */
    dg->momentum_space = 0;

    /* We did allocate buffers */
    dg->buffer_width = buffer_width;

    return 0;
}

int free_local_grid(struct distributed_grid *dg) {
    free_local_real_grid(dg);
    free_local_complex_grid(dg);
    return 0;
}

int free_local_real_grid(struct distributed_grid *dg) {
    if (dg->buffer_width > 0) {
        fft_free(dg->buffered_box);
    } else {
        fft_free(dg->box);
    }
    return 0;
}

int free_local_complex_grid(struct distributed_grid *dg) {
#ifndef USE_IN_PLACE_FFTS
    fft_free(dg->fbox);
#endif
    return 0;
}

int alloc_local_buffers(struct distributed_grid *dg, int buffer_width) {
    /* Check that the buffer sizes are valid */
    if (buffer_width > dg->NX) {
        printf("There are too many MPI ranks - increase the grid size!\n");
        printf("Additionally, make sure that they are integer multiples.\n");
        exit(1);
    }

    dg->buffer_width = buffer_width;
    dg->buffer_left = fft_alloc_real(buffer_width * dg->N * dg->N);
    dg->buffer_right = fft_alloc_real(buffer_width * dg->N * dg->N);
    return 0;
}

int free_local_buffers(struct distributed_grid *dg) {
    fft_free(dg->buffer_left);
    fft_free(dg->buffer_right);
    return 0;
}

/* Make copies of the edges of the local grids and communicate them to
 * neighbouring MPI ranks. Assumes memory has already been allocated. */
int create_local_buffers(struct distributed_grid *dg) {
    /* Get the dimensions of the cluster */
    int rank, MPI_Rank_Count;
    MPI_Comm_rank(dg->comm, &rank);
    MPI_Comm_size(dg->comm, &MPI_Rank_Count);

    /* Is there anything to do? */
    if (MPI_Rank_Count < 2) return 0;

    /* The MPI ranks are placed along a periodic ring */
    int rank_left = (rank == 0) ? MPI_Rank_Count - 1 : rank - 1;
    int rank_right = (rank + 1) % MPI_Rank_Count;

    /* Sizes of the buffers */
    long int size = dg->buffer_width * dg->Ny * dg->Nz;

    /* The first element for the right buffer of the left neighbour */
    long int first_left = 0;
    /* The first element for the left buffer of the right neighbour */
    long int first_right = (dg->NX - dg->buffer_width) * dg->Ny * dg->Nz;

    /* Send buffers left and right, using non-blocking calls */
    MPI_Request delivery_left;
    MPI_Request delivery_right;
    MPI_Isend(dg->box + first_left, size, MPI_GRID_TYPE, rank_left, 2, MPI_COMM_WORLD, &delivery_left);
    MPI_Isend(dg->box + first_right, size, MPI_GRID_TYPE, rank_right, 2, MPI_COMM_WORLD, &delivery_right);

    /* Probe and receive buffers from the left and right when ready */
    int finished_left = 0, finished_right = 0;
    while (!finished_left || !finished_right) {
        /* Probe and receive left, blocking only when ready */
        if (!finished_left) {
            int ready_left = 0;
            MPI_Status status_left;
            MPI_Iprobe(rank_left, 2, MPI_COMM_WORLD, &ready_left, &status_left);
            if (ready_left) {
                MPI_Recv(dg->buffer_left, size, MPI_GRID_TYPE, rank_left, 2,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                finished_left = 1;
            }
        }

        /* Probe and receive right, blocking only when ready */
        if (!finished_right) {
            int ready_right = 0;
            MPI_Status status_right;
            MPI_Iprobe(rank_right, 2, MPI_COMM_WORLD, &ready_right, &status_right);
            if (ready_right) {
                MPI_Recv(dg->buffer_right, size, MPI_GRID_TYPE, rank_right, 2,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                finished_right = 1;
            }
        }
    }

    MPI_Wait(&delivery_left, MPI_STATUS_IGNORE);
    MPI_Wait(&delivery_right, MPI_STATUS_IGNORE);

    return 0;
}

/* Send the buffers to the neighbour and add the contributions to the main grid.*/
int add_local_buffers(struct distributed_grid *dg) {
    /* Get the dimensions of the cluster */
    int rank, MPI_Rank_Count;
    MPI_Comm_rank(dg->comm, &rank);
    MPI_Comm_size(dg->comm, &MPI_Rank_Count);

    /* Is there anything to do? */
    if (MPI_Rank_Count < 2) return 0;

    /* The MPI ranks are placed along a periodic ring */
    int rank_left = (rank == 0) ? MPI_Rank_Count - 1 : rank - 1;
    int rank_right = (rank + 1) % MPI_Rank_Count;

    /* Sizes of the buffers */
    long long int size = dg->buffer_width * dg->Ny * dg->Nz;

    /* Allocate temporary arrays for the buffers */
    GridFloatType *recv_buffer_left = fft_alloc_real(size);
    GridFloatType *recv_buffer_right = fft_alloc_real(size);

    /* Send buffers left and right, using non-blocking calls */
    MPI_Request delivery_left;
    MPI_Request delivery_right;
    MPI_Isend(dg->buffer_left, size, MPI_GRID_TYPE, rank_left, 1, MPI_COMM_WORLD, &delivery_left);
    MPI_Isend(dg->buffer_right, size, MPI_GRID_TYPE, rank_right, 1, MPI_COMM_WORLD, &delivery_right);

    /* Probe and receive buffers from the left and right when ready */
    int finished_left = 0, finished_right = 0;
    while (!finished_left || !finished_right) {
        /* Probe and receive left, blocking only when ready */
        if (!finished_left) {
            int ready_left = 0;
            MPI_Status status_left;
            MPI_Iprobe(rank_left, 1, MPI_COMM_WORLD, &ready_left, &status_left);
            if (ready_left) {
                MPI_Recv(recv_buffer_left, size, MPI_GRID_TYPE, rank_left, 1,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                /* Add the left buffer to the main grid */
                long int first_left = 0;
                for (long int i = 0; i < size; i ++) {
                    dg->box[first_left + i] += recv_buffer_left[i];
                }
                finished_left = 1;
            }
        }

        /* Probe and receive right, blocking only when ready */
        if (!finished_right) {
            int ready_right = 0;
            MPI_Status status_right;
            MPI_Iprobe(rank_right, 1, MPI_COMM_WORLD, &ready_right, &status_right);
            if (ready_right) {
                MPI_Recv(recv_buffer_right, size, MPI_GRID_TYPE, rank_right, 1,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                /* Add the right buffer to the main grid */
                long int first_right = (dg->NX - dg->buffer_width) * dg->Ny * dg->Nz;
                for (long int i = 0; i < size; i ++) {
                    dg->box[first_right + i] += recv_buffer_right[i];
                }
                finished_right = 1;
            }
        }
    }

    MPI_Wait(&delivery_left, MPI_STATUS_IGNORE);
    MPI_Wait(&delivery_right, MPI_STATUS_IGNORE);

    /* Free the temporary arrays */
    fft_free(recv_buffer_left);
    fft_free(recv_buffer_right);

    return 0;
}
