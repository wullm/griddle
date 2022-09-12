/*******************************************************************************
 * This file is part of Nyver.
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

int alloc_local_grid(struct distributed_grid *dg, int N, double boxlen, MPI_Comm comm) {
    /* Determine the size of the local portion */
    dg->local_size = fftw_mpi_local_size_3d(N, N, N/2+1, comm, &dg->NX, &dg->X0);

    /* Store a reference to the communicator */
    dg->comm = comm;

    /* Store basic attributes */
    dg->N = N;
    dg->boxlen = boxlen;

    /* Allocate memory for the complex and real arrays */
    dg->fbox = fftw_alloc_complex(dg->local_size);
    dg->box = fftw_alloc_real(2*dg->local_size);

    /* This flag will be flipped each time we do a Fourier transform */
    dg->momentum_space = 0;

    return 0;
}

int free_local_grid(struct distributed_grid *dg) {
    free_local_real_grid(dg);
    free_local_complex_grid(dg);
    return 0;
}

int free_local_real_grid(struct distributed_grid *dg) {
    fftw_free(dg->box);
    return 0;
}

int free_local_complex_grid(struct distributed_grid *dg) {
    fftw_free(dg->fbox);
    return 0;
}
