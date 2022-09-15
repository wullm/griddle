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
#include <mpi.h>

#include "../include/particle_exchange.h"
#include "../include/message.h"

/* Check that all particles are on the right rank */
long long int count_foreign_particles(struct particle *parts, double boxlen,
                                      long long int num_localpart) {

    /* Get the dimensions of the cluster */
    int rank, MPI_Rank_Count;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &MPI_Rank_Count);

    /* Count the number of local particles that belong on another MPI rank */
    long long int foreign_particles = 0;
    for (long long i = 0; i < num_localpart; i++) {
        struct particle *p = &parts[i];
        if (p->rank != rank) foreign_particles++;
    }

    /* Communicate amongst all ranks */
    long long int total_foreign_parts;
    MPI_Allreduce(&foreign_particles, &total_foreign_parts, 1, MPI_LONG_LONG,
                  MPI_SUM, MPI_COMM_WORLD);

    return total_foreign_parts;
}

/* Exchange particles between MPI ranks */
int exchange_particles(struct particle *parts, double boxlen,
                       long long int *num_localpart, int iteration) {
    /* Get the dimensions of the cluster */
    int rank, MPI_Rank_Count;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &MPI_Rank_Count);

    /* Is there anything to do? */
    if (MPI_Rank_Count <= 1) {
        return 0;
    }

    /* Count the number of local particles that belong on each MPI rank */
    long long int *rank_num_parts = calloc(MPI_Rank_Count, sizeof(long long int));
    for (long long i = 0; i < *num_localpart; i++) {
        struct particle *p = &parts[i];

        int on_rank = (int) ((p->x[0] / boxlen) * MPI_Rank_Count);
        rank_num_parts[on_rank]++;

        p->rank = on_rank;
    }

    /* Sort particles by their desired MPI rank */
    qsort(parts, *num_localpart, sizeof(struct particle), particleSort);

    /* The MPI ranks are placed along a periodic ring */
    int rank_left = (rank == 0) ? MPI_Rank_Count - 1 : rank - 1;
    int rank_right = (rank + 1) % MPI_Rank_Count;

    /* Decide whether particles should be sent left or right */
    long long int num_send_left = 0;
    long long int num_send_right = 0;
    long long int first_send_left = INT64_MAX - 1; // = infinity
    long long int first_send_right = INT64_MAX - 1; // = infinity
    for (long long i = 0; i < *num_localpart; i++) {
        struct particle *p = &parts[i];

        if (p->rank != rank) {
            if (abs(p->rank - rank_left) < abs(p->rank - rank_right)) {
                num_send_left++;
                if (i < first_send_left) first_send_left = i;
            } else {
                num_send_right++;
                if (i < first_send_right) first_send_right = i;
            }
        }
    }

    /** **/
    /* First, send particles to the right */
    /** **/

    /* Communicate the number of particles to be received */
    long long int receive_from_left;
    if (rank > 0) {
        MPI_Recv(&receive_from_left, 1, MPI_LONG_LONG, rank_left, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    MPI_Send(&num_send_right, 1, MPI_LONG_LONG, rank_right,
             0, MPI_COMM_WORLD);
    if (rank == 0) {
        MPI_Recv(&receive_from_left, 1, MPI_LONG_LONG, rank_left,
                 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    /* Allocate memory for particles to be received */
    struct particle *receive_parts_left = malloc(receive_from_left * sizeof(struct particle));

    if (rank > 0) {
        MPI_Recv(receive_parts_left, receive_from_left * sizeof(struct particle),
                 MPI_CHAR, rank_left, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    MPI_Send(&parts[first_send_right], num_send_right * sizeof(struct particle),
             MPI_CHAR, rank_right, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        MPI_Recv(receive_parts_left, receive_from_left * sizeof(struct particle),
                 MPI_CHAR, rank_left, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    }

    /** **/
    /* Next, send particles to the left */
    /** **/

    /* Communicate the number of particles to be received */
    long long int receive_from_right;
    if (rank > 0) {
        MPI_Recv(&receive_from_right, 1, MPI_LONG_LONG, rank_right, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    MPI_Send(&num_send_left, 1, MPI_LONG_LONG, rank_left,
             0, MPI_COMM_WORLD);
    if (rank == 0) {
        MPI_Recv(&receive_from_right, 1, MPI_LONG_LONG, rank_right,
                 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    /* Allocate memory for particles to be received */
    struct particle *receive_parts_right = malloc(receive_from_right * sizeof(struct particle));

    if (rank < MPI_Rank_Count - 1) {
        MPI_Recv(receive_parts_right, receive_from_right * sizeof(struct particle),
                 MPI_CHAR, rank_right, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    MPI_Send(&parts[first_send_left], num_send_left * sizeof(struct particle),
             MPI_CHAR, rank_left, 0, MPI_COMM_WORLD);
    if (rank == MPI_Rank_Count - 1) {
        MPI_Recv(receive_parts_right, receive_from_right * sizeof(struct particle),
                 MPI_CHAR, rank_right, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    }

    /* Move data around, overwriting particles that were sent away */
    if (rank == 0) {
        /* Insert the particles received from the right */
        memmove(parts + *num_localpart - num_send_left - num_send_right,
                receive_parts_right, receive_from_right * sizeof(struct particle));
        /* Insert the particles received from the left */
        memmove(parts + *num_localpart - num_send_left - num_send_right + receive_from_right,
                receive_parts_left, receive_from_left * sizeof(struct particle));
    } else if (rank > 0 && rank < MPI_Rank_Count - 1) {
        /* Make space for particles on the left */
        memmove(parts + receive_from_left, parts + num_send_left,
                (*num_localpart - num_send_left) * sizeof(struct particle));
        /* Insert the particles received from the left */
        memmove(parts, receive_parts_left, receive_from_left * sizeof(struct particle));
        /* Insert the particles received from the right at the end */
        memmove(parts + *num_localpart - num_send_left - num_send_right + receive_from_left,
                receive_parts_right, receive_from_right * sizeof(struct particle));
    } else {
        /* Make space for particles on the left */
        memmove(parts + receive_from_left + receive_from_right,
                parts + num_send_left + num_send_right,
                (*num_localpart - num_send_left - num_send_right) * sizeof(struct particle));
        /* Insert the particles received from the left */
        memmove(parts, receive_parts_left, receive_from_left * sizeof(struct particle));
        /* Insert the particles received from the right */
        memmove(parts + receive_from_left,
                receive_parts_right, receive_from_right * sizeof(struct particle));
    }

    /* Update the particle numbers */
    *num_localpart = *num_localpart - num_send_left + receive_from_left - num_send_right  + receive_from_right;

    /* Free memory used for receiving particle data */
    free(receive_parts_left);
    free(receive_parts_right);

    /* Free particle rank count array */
    free(rank_num_parts);

    /* Check that everything is now where it should be */
    long long int total_foreign_parts = count_foreign_particles(parts, boxlen, *num_localpart);
    if (total_foreign_parts > 0) {
        iteration++;

        message(rank, "Exchanging: %lld foreign particles remain after %d iterations.\n", total_foreign_parts, iteration);

        if (iteration < MPI_Rank_Count + 1) {
            exchange_particles(parts, boxlen, num_localpart, iteration);
        } else {
            printf("Maximum number of iterations exceeded.\n");
            exit(1);
        }
    }

    return 0;
}
