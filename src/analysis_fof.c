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

#include "../include/analysis_fof.h"
#include "../include/message.h"

#define DEBUG_CHECKS

static inline int row_major_cell(int i, int j, int k, int N_cells) {
    return i * N_cells * N_cells + j * N_cells + k;
}

/* Determine the cell containing a given particle */
static inline int which_cell(IntPosType x[3], double int_to_cell_fac, int N_cells) {
    return row_major_cell((int) (int_to_cell_fac * x[0]), (int) (int_to_cell_fac * x[1]), (int) (int_to_cell_fac * x[2]), N_cells);
}

/* Order particles by their spatial cell index */
static inline int cellListSort(const void *a, const void *b) {
    struct fof_cell_list *ca = (struct fof_cell_list*) a;
    struct fof_cell_list *cb = (struct fof_cell_list*) b;

    return ca->cell >= cb->cell;
}

/* Should two particles be linked? */
static inline int should_link(const IntPosType ax[3], const IntPosType bx[3],
                              double int_to_pos_fac, double linking_length_2) {

    /* Vector distance */
    const IntPosType dx = bx[0] - ax[0];
    const IntPosType dy = bx[1] - ax[1];
    const IntPosType dz = bx[2] - ax[2];

    /* Enforce boundary conditions */
    const IntPosType tx = (dx < -dx) ? dx : -dx;
    const IntPosType ty = (dy < -dy) ? dy : -dy;
    const IntPosType tz = (dz < -dz) ? dz : -dz;

    /* Convert to physical lengths */
    const double fx = tx * int_to_pos_fac;
    const double fy = ty * int_to_pos_fac;
    const double fz = tz * int_to_pos_fac;

    const double r_2 = fx * fx + fy * fy + fz * fz;

    return r_2 <= linking_length_2;
}


/* Find the root of the set of a given particle */
long int find_root(struct fof_part_data *fof_parts, struct fof_part_data *part) {
    if (part->local_offset == part->root)
        return part->local_offset;
    part->root = find_root(fof_parts, &fof_parts[part->root]);
    return part->root;
}

/* Perform the union operation on the sets containing two given particles */
void union_roots(struct fof_part_data *fof_parts, struct fof_part_data *a, struct fof_part_data *b) {

    long int root_a = find_root(fof_parts, a);
    long int root_b = find_root(fof_parts, b);

    if (root_a != root_b) {
        if (fof_parts[root_a].global_offset <= fof_parts[root_b].global_offset) {
            fof_parts[root_a].root = root_b;
        } else {
            fof_parts[root_b].root = root_a;
        }
    }
}

/* Link particles within two cells */
int link_cells(struct fof_part_data *fof_parts, struct fof_cell_list *cl,
               long int local_offset1, long int local_offset2,
               long int local_count1, long int local_count2,
               double int_to_pos_fac, double linking_length_2) {

    if (local_count1 < 1 || local_count2 < 1 || (local_count1 + local_count2 < 2)) return 0;

    int links = 0;

    for (int a = 0; a < local_count1; a++) {
        const int index_a = cl[local_offset1 + a].offset;
        const IntPosType *xa = fof_parts[index_a].x;

        /* If we are linking within the same cell, only check all pairs once */
        int max_check = (local_offset1 == local_offset2) ? a : local_count2;

        for (int b = 0; b < max_check; b++) {
            const int index_b = cl[local_offset2 + b].offset;
            const IntPosType *xb = fof_parts[index_b].x;
            if (should_link(xa, xb, int_to_pos_fac, linking_length_2)) {
                links++;
                union_roots(fof_parts, &fof_parts[index_a], &fof_parts[index_b]);
            }
        }
    }

    return links;
}

/* TODO, kick and drift particles to the right time */
int analysis_fof(struct particle *parts, double boxlen, long long int Ng,
                 long long int num_localpart, long long int max_partnum,
                 double linking_length, int halo_min_npart, int output_num,
                 double a_scale_factor) {

    /* Get the dimensions of the cluster */
    int rank, MPI_Rank_Count;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &MPI_Rank_Count);

    /* The MPI ranks are placed along a periodic ring */
    int rank_left = (rank == 0) ? MPI_Rank_Count - 1 : rank - 1;
    int rank_right = (rank + 1) % MPI_Rank_Count;

    /* Data type for MPI communication of particles */
    MPI_Datatype fof_type = mpi_fof_data_type();

    /* Communicate the particle counts across all ranks */
    long long int *parts_per_rank = malloc(MPI_Rank_Count * sizeof(long int));
    long long int *rank_offsets = malloc(MPI_Rank_Count * sizeof(long int));

    MPI_Allgather(&num_localpart, 1, MPI_LONG_LONG, parts_per_rank,
                  1, MPI_LONG_LONG, MPI_COMM_WORLD);

    rank_offsets[0] = 0;
    for (int i = 1; i < MPI_Rank_Count; i++) {
        rank_offsets[i] = rank_offsets[i - 1] + parts_per_rank[i - 1];
    }

    /* The local rank offset */
    const long int rank_offset = rank_offsets[rank];

    /* Timer */
    struct timepair fof_timer;
    timer_start(rank, &fof_timer);

    /* The initial domain decomposition into spatial cells */
    const int N_cells = boxlen / (4.0 * linking_length);
    const double int_to_cell_fac = N_cells / pow(2.0, POSITION_BITS);

    if (N_cells > 1250) {
        printf("The number of cells is large. We should switch to larger ints (TODO).\n");
        exit(1);
    }

    /* The conversion factor from integers to physical lengths */
    const double pos_to_int_fac = pow(2.0, POSITION_BITS) / boxlen;
    const double int_to_pos_fac = 1.0 / pos_to_int_fac;
    const double linking_length_2 = linking_length * linking_length;

    /* The conversion factor from integers to MPI rank number */
    const long long int max_block_width = Ng / MPI_Rank_Count + ((Ng % MPI_Rank_Count) ? 1 : 0); //rounded up
    const double int_block_width = max_block_width * (boxlen / Ng * pos_to_int_fac);
    const double int_to_rank_fac = 1.0 / int_block_width;
    const double rank_to_pos_fac = int_to_pos_fac / int_to_rank_fac;

    /* The buckets must be larger than the linking length */
    if (boxlen / N_cells <= 2 * linking_length) {
        printf("The spatial buckets are smaller than the linking length, which is bad.\n");
        exit(1);
    }

    /* The ranks must cover a physical length larger than the linking length */
    if (boxlen / MPI_Rank_Count <= 2 * linking_length) {
        printf("The number of ranks is so high that they do not span at least one linking length!\n");
        exit(1);
    }

    /* Copy over the hot particle data */
    struct fof_part_data *fof_parts = malloc(max_partnum * sizeof(struct fof_part_data));

    for (long int i = 0; i < num_localpart; i++) {
        fof_parts[i].x[0] = parts[i].x[0];
        fof_parts[i].x[1] = parts[i].x[1];
        fof_parts[i].x[2] = parts[i].x[2];
        fof_parts[i].global_offset = i + rank_offset;
        // root and local_offset set later
    }

    timer_stop(rank, &fof_timer, "Copying particle data took ");

    /* The total number of particles to be received from the right */
    int receive_foreign_count = 0;
    int receive_from_right = 0;

    if (MPI_Rank_Count > 1) {

        /* Count the number of particles within one linking length from the left edge */
        int count_near_edge = 0;
        for (long int i = 0; i < num_localpart; i++) {
            double rank_float = fof_parts[i].x[0] * int_to_rank_fac;
            double dx = rank_float - ((int) rank_float);
            double dx_phys = dx * rank_to_pos_fac;
            if (dx_phys < linking_length) {
                count_near_edge++;
            }
        }

        /* Fish out particles within one linking length from the left edge */
        struct fof_part_data *edge_parts = malloc(count_near_edge * sizeof(struct fof_part_data));
        int copy_counter = 0;
        for (long int i = 0; i < num_localpart; i++) {
            double rank_float = fof_parts[i].x[0] * int_to_rank_fac;
            double dx = rank_float - ((int) rank_float);
            double dx_phys = dx * rank_to_pos_fac;
            if (dx_phys < linking_length) {
                memcpy(edge_parts + copy_counter, fof_parts + i, sizeof(struct fof_part_data));
                copy_counter++;
            }
        }

        timer_stop(rank, &fof_timer, "Copying edge particles took ");

        if (rank == MPI_Rank_Count - 1) {
            /* Communicate the edge particles to the left neighbour */
            MPI_Send(edge_parts, count_near_edge, fof_type, rank_left, 0, MPI_COMM_WORLD);

            /* Release the delivered particles */
            free(edge_parts);

            /* In this special case, also count the number of particles within
             * one linking length from the right edge */
            int count_near_right_edge = 0;
            for (long int i = 0; i < num_localpart; i++) {
                double pos = fof_parts[i].x[0] * int_to_pos_fac;
                double dx_phys = boxlen - pos;
                if (dx_phys < linking_length) {
                    count_near_right_edge++;
                }
            }

            /* Fish out particles within one linking length from the left edge */
            edge_parts = malloc(count_near_right_edge * sizeof(struct fof_part_data));
            copy_counter = 0;
            for (long int i = 0; i < num_localpart; i++) {
                double pos = fof_parts[i].x[0] * int_to_pos_fac;
                double dx_phys = boxlen - pos;
                if (dx_phys < linking_length) {
                    memcpy(edge_parts + copy_counter, fof_parts + i, sizeof(struct fof_part_data));
                    copy_counter++;
                }
            }

            /* Communicate these edge particles to the right neighbour */
            MPI_Send(edge_parts, count_near_right_edge, fof_type, rank_right, 0, MPI_COMM_WORLD);

            /* Release the delivered particles */
            free(edge_parts);
        } else if (rank > 0) {
            /* Prepare to receive particles from the right */
            MPI_Status status_right;
            MPI_Probe(rank_right, 0, MPI_COMM_WORLD, &status_right);
            MPI_Get_count(&status_right, fof_type, &receive_from_right);
            receive_foreign_count += receive_from_right;

            /* Check that we have enough memory */
            if (num_localpart + receive_foreign_count > max_partnum) {
                printf("Not enough memory to exchange particles on rank %d (%lld < %lld).\n", rank, max_partnum, num_localpart + receive_foreign_count);
                exit(1);
            }

            /* Receive the particle data */
            struct fof_part_data *receive_parts = malloc(receive_from_right * sizeof(struct fof_part_data));
            MPI_Recv(receive_parts, receive_from_right, fof_type,
                     rank_right, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            /* Move the received particles to the end of the FOF particle array */
            memcpy(fof_parts + num_localpart, receive_parts, receive_from_right * sizeof(struct fof_part_data));

            /* Release the received particles */
            free(receive_parts);

            /* Communicate the edge particles to the left neighbour */
            MPI_Send(edge_parts, count_near_edge, fof_type, rank_left, 0, MPI_COMM_WORLD);

            /* Release the delivered particles */
            free(edge_parts);
        }

        /* Finally, receive on rank 0 - in this case from both sides */
        else if (rank == 0) {
            /* Prepare to receive particles from the right */
            MPI_Status status_right;
            MPI_Probe(rank_right, 0, MPI_COMM_WORLD, &status_right);
            MPI_Get_count(&status_right, fof_type, &receive_from_right);
            receive_foreign_count += receive_from_right;

            /* Check that we have enough memory */
            if (num_localpart + receive_foreign_count > max_partnum) {
                printf("Not enough memory to exchange particles on rank %d (%lld < %lld).\n", rank, max_partnum, num_localpart + receive_foreign_count);
                exit(1);
            }

            /* Receive the particle data */
            struct fof_part_data *receive_parts = malloc(receive_from_right * sizeof(struct fof_part_data));
            MPI_Recv(receive_parts, receive_from_right, fof_type,
                     rank_right, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            /* Move the received particles to the end of the FOF particle array */
            memcpy(fof_parts + num_localpart, receive_parts, receive_from_right * sizeof(struct fof_part_data));

            /* Release the received particles */
            free(receive_parts);

            /* Next, prepare to receive particles from the left */
            int receive_from_left_0;
            MPI_Status status_left;
            MPI_Probe(rank_left, 0, MPI_COMM_WORLD, &status_left);
            MPI_Get_count(&status_left, fof_type, &receive_from_left_0);
            receive_foreign_count += receive_from_left_0;

            /* Check that we have enough memory */
            if (num_localpart + receive_foreign_count > max_partnum) {
                printf("Not enough memory to exchange particles on rank %d (%lld < %lld).\n", rank, max_partnum, num_localpart + receive_foreign_count);
                exit(1);
            }

            /* Receive the particle data */
            receive_parts = malloc(receive_from_left_0 * sizeof(struct fof_part_data));
            MPI_Recv(receive_parts, receive_from_left_0, fof_type,
                     rank_left, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            /* Move the received particles to the end of the FOF particle array */
            memcpy(fof_parts + num_localpart + receive_from_right, receive_parts, receive_from_left_0 * sizeof(struct fof_part_data));

            /* Release the received particles */
            free(receive_parts);
        }

        timer_stop(rank, &fof_timer, "The first communication took ");
    }

    /* Set the local offsets for the local and foreign particles */
    for (long long i = 0; i < num_localpart + receive_foreign_count; i++) {
        fof_parts[i].local_offset = i;
        fof_parts[i].root = i;
    }

    /* Now create the cell-particle list */
    struct fof_cell_list *cell_list = malloc((num_localpart + receive_foreign_count) * sizeof(struct fof_cell_list));
    for (long long i = 0; i < num_localpart + receive_foreign_count; i++) {
        cell_list[i].cell = which_cell(fof_parts[i].x, int_to_cell_fac, N_cells);
        cell_list[i].offset = i;
    }

    /* Sort particles into spatial cells */
    qsort(cell_list, num_localpart + receive_foreign_count, sizeof(struct fof_cell_list), cellListSort);

#ifdef DEBUG_CHECKS
    /* Check the sort */
    for (long long i = 1; i < num_localpart + receive_foreign_count; i++) {
        assert(cell_list[i].cell >= cell_list[i - 1].cell);
    }
#endif

    timer_stop(rank, &fof_timer, "Sorting particles took ");

    /* Determine the counts and offsets of particles in each cell */
    const int num_cells = N_cells * N_cells * N_cells;
    long int *cell_counts = calloc(num_cells, sizeof(long int));
    long int *cell_offsets = calloc(num_cells, sizeof(long int));

    /* Count particles in cells */
    for (long long i = 0; i < num_localpart + receive_foreign_count; i++) {
        int c = cell_list[i].cell;
#ifdef DEBUG_CHECKS
        assert((c >= 0) && (c < num_cells));
#endif
        cell_counts[c]++;
    }

    /* Determine the offsets, using the fact that the particles are sorted */
    cell_offsets[0] = 0;
    for (int i = 1; i < num_cells; i++) {
        cell_offsets[i] = cell_offsets[i-1] + cell_counts[i-1];
    }

    message(rank, "Linking particles in cells.\n");

    long int total_links = 0;

    /* Now link across cells */
    for (int i = 0; i < N_cells; i++) {
        for (int j = 0; j < N_cells; j++) {
            for (int k = 0; k < N_cells; k++) {

                long int offset = cell_offsets[row_major_cell(i, j, k, N_cells)];
                long int count = cell_counts[row_major_cell(i, j, k, N_cells)];

                int i1 = i + 1;
                if (i1 >= N_cells) i1 -= N_cells;
                int j1 = j + 1;
                if (j1 >= N_cells) j1 -= N_cells;
                int k1 = k + 1;
                if (k1 >= N_cells) k1 -= N_cells;

                long int offset1 = cell_offsets[row_major_cell(i1, j, k, N_cells)];
                long int offset2 = cell_offsets[row_major_cell(i, j1, k, N_cells)];
                long int offset3 = cell_offsets[row_major_cell(i, j, k1, N_cells)];
                long int offset4 = cell_offsets[row_major_cell(i1, j1, k, N_cells)];
                long int offset5 = cell_offsets[row_major_cell(i, j1, k1, N_cells)];
                long int offset6 = cell_offsets[row_major_cell(i1, j, k1, N_cells)];
                long int offset7 = cell_offsets[row_major_cell(i1, j1, k1, N_cells)];
                long int count1 = cell_counts[row_major_cell(i1, j, k, N_cells)];
                long int count2 = cell_counts[row_major_cell(i, j1, k, N_cells)];
                long int count3 = cell_counts[row_major_cell(i, j, k1, N_cells)];
                long int count4 = cell_counts[row_major_cell(i1, j1, k, N_cells)];
                long int count5 = cell_counts[row_major_cell(i, j1, k1, N_cells)];
                long int count6 = cell_counts[row_major_cell(i1, j, k1, N_cells)];
                long int count7 = cell_counts[row_major_cell(i1, j1, k1, N_cells)];

                /* Cell self-links */
                total_links += link_cells(fof_parts, cell_list, offset, offset, count, count, int_to_pos_fac, linking_length_2);

                /* Cell-cell neighbour links */
                total_links += link_cells(fof_parts, cell_list, offset, offset1, count, count1, int_to_pos_fac, linking_length_2);
                total_links += link_cells(fof_parts, cell_list, offset, offset2, count, count2, int_to_pos_fac, linking_length_2);
                total_links += link_cells(fof_parts, cell_list, offset, offset3, count, count3, int_to_pos_fac, linking_length_2);
                total_links += link_cells(fof_parts, cell_list, offset, offset4, count, count4, int_to_pos_fac, linking_length_2);
                total_links += link_cells(fof_parts, cell_list, offset, offset5, count, count5, int_to_pos_fac, linking_length_2);
                total_links += link_cells(fof_parts, cell_list, offset, offset6, count, count6, int_to_pos_fac, linking_length_2);
                total_links += link_cells(fof_parts, cell_list, offset, offset7, count, count7, int_to_pos_fac, linking_length_2);
            }
        }
    }

    timer_stop(rank, &fof_timer, "Linking particles in cells took ");
    message(rank, "Found %ld links within and across cells on rank %d.\n", total_links, rank);

    /* Loop over the roots again to collapse the tree */
    for (long int i = 0; i < num_localpart + receive_foreign_count; i++) {
        fof_parts[i].root = find_root(fof_parts, &fof_parts[i]);
    }

    /* Compute the global offsets of the roots */
    long int *global_roots = malloc((num_localpart + receive_foreign_count) * sizeof(long int));
    for (long int i = 0; i < num_localpart + receive_foreign_count; i++) {
        global_roots[i] = fof_parts[fof_parts[i].root].global_offset;
    }

    /* Overwrite the roots by the global roots */
    for (long int i = 0; i < num_localpart + receive_foreign_count; i++) {
        fof_parts[i].root = global_roots[i];
    }

    /* Free the global root array */
    free(global_roots);

    /* Communicate the particles with foreign roots to the reverse neighbour rank */
    struct fof_part_data *returned_parts = NULL;

    /* The total number of particles to be received from the left */
    int receive_from_left = 0;

    if (MPI_Rank_Count > 1) {

        if (rank == 0) {
            /* Determine the number of local particles with foreign roots */
            int num_foreign_rooted = 0;
            for (long int i = 0; i < num_localpart; i++) {
                if (fof_parts[i].root < rank_offset || fof_parts[i].root >= rank_offset + num_localpart) {
                    num_foreign_rooted++;
                }
            }

            /* Now, fish out all local particles with foreign roots */
            long int copy_counter = 0;
            struct fof_part_data *foreign_root_parts = malloc(num_foreign_rooted * sizeof(struct fof_part_data));
            for (long int i = 0; i < num_localpart; i++) {
                if (fof_parts[i].root < rank_offset || fof_parts[i].root >= rank_offset + num_localpart) {
                    memcpy(foreign_root_parts + copy_counter, fof_parts + i, sizeof(struct fof_part_data));
                    copy_counter++;
                }
            }

            /* Disable the foreign rooted particles on this rank by turning them into singletons */
            for (long int i = 0; i < num_localpart; i++) {
                if (fof_parts[i].root < rank_offset || fof_parts[i].root >= rank_offset + num_localpart) {
                    fof_parts[i].root = fof_parts[i].global_offset;
                }
            }

            /* Communicate the foreign particles to the right neighbour */
            MPI_Send(foreign_root_parts, num_foreign_rooted, fof_type, rank_right, 0, MPI_COMM_WORLD);

            /* Release the delivered particles */
            free(foreign_root_parts);
        } else {
            /* Prepare to receive particles from the left */
            MPI_Status status_left;
            MPI_Probe(rank_left, 0, MPI_COMM_WORLD, &status_left);
            MPI_Get_count(&status_left, fof_type, &receive_from_left);

            /* Receive the particle data */
            returned_parts = malloc(receive_from_left * sizeof(struct fof_part_data));
            MPI_Recv(returned_parts, receive_from_left, fof_type,
                     rank_left, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            /* Check that we have enough memory */
            if (num_localpart + receive_from_left > max_partnum) {
                printf("Not enough memory to exchange particles on rank %d (%lld < %lld).\n", rank, max_partnum, num_localpart + receive_from_left);
                exit(1);
            }

            /* Move to received particles into the main memory */
            memcpy(fof_parts + num_localpart, returned_parts, receive_from_left * sizeof(struct fof_part_data));

            /* Release the received particles */
            free(returned_parts);

            /* Determine the number of particles with foreign roots (that are not roots themselves) */
            int num_foreign_rooted = 0;
            for (long int i = 0; i < num_localpart + receive_from_left; i++) {
                /* Skip duplicate domestic particles */
                if (((fof_parts[i].x[0] * int_to_rank_fac) != rank || i < num_localpart) && fof_parts[i].root != fof_parts[i].global_offset && (fof_parts[i].root < rank_offset || fof_parts[i].root >= rank_offset + num_localpart)) {
                    num_foreign_rooted++;
                }
            }

            /* Now, fish out all local particles with foreign roots */
            long int copy_counter = 0;
            struct fof_part_data *foreign_root_parts = malloc(num_foreign_rooted * sizeof(struct fof_part_data));
            for (long int i = 0; i < num_localpart + receive_from_left; i++) {
                if (((fof_parts[i].x[0] * int_to_rank_fac) != rank || i < num_localpart) && fof_parts[i].root != fof_parts[i].global_offset && (fof_parts[i].root < rank_offset || fof_parts[i].root >= rank_offset + num_localpart)) {
                    memcpy(foreign_root_parts + copy_counter, fof_parts + i, sizeof(struct fof_part_data));
                    copy_counter++;
                }
            }

            /* Disable the foreign rooted particles on this rank by turning them into singletons */
            for (long int i = 0; i < num_localpart + receive_from_left; i++) {
                if (((fof_parts[i].x[0] * int_to_rank_fac) != rank || i < num_localpart) && fof_parts[i].root != fof_parts[i].global_offset && (fof_parts[i].root < rank_offset || fof_parts[i].root >= rank_offset + num_localpart)) {
                    fof_parts[i].root = fof_parts[i].global_offset;
                }
            }

            /* Communicate the foreign particles to the right neighbour */
            if (rank < MPI_Rank_Count - 1) {
                MPI_Send(foreign_root_parts, num_foreign_rooted, fof_type, rank_right, 0, MPI_COMM_WORLD);
            }

            /* Release the delivered particles */
            free(foreign_root_parts);
        }

        timer_stop(rank, &fof_timer, "The second communication took ");
        MPI_Barrier(MPI_COMM_WORLD);
    }

    /* Determine the number of foreign roots of local particles */
    int foreign_rooted = 0;
    for (long int i = 0; i < num_localpart + receive_from_left; i++) {
        if (fof_parts[i].root != fof_parts[i].global_offset && (fof_parts[i].root < rank_offset || fof_parts[i].root >= rank_offset + num_localpart)) {
            foreign_rooted++;
        }
    }

    if (foreign_rooted > 0) {
        printf("We still have particles with foreign roots. Need to do more communications.\n");
        exit(1);
    }

    /* Now that we have no foreign rooted particles, we can attach the trees */

    /* For the local particles, turn the global_roots back into roots */
    for (long long i = 0; i < num_localpart; i++) {
        fof_parts[i].root -= rank_offset;
    }

    /* Attach returned particles to the local tree using the proper local offsets */
    for (long long i = num_localpart; i < num_localpart + receive_from_left; i++) {
        /* Skip foreign singletons */
        if (i >= num_localpart && fof_parts[i].root == fof_parts[i].global_offset) continue;
        /* Skip duplicate domestic particles */
        if (i >= num_localpart && (fof_parts[i].x[0] * int_to_rank_fac) == rank) continue;

        fof_parts[i].local_offset = i;
        fof_parts[i].root = find_root(fof_parts, &fof_parts[fof_parts[i].root - rank_offset]);
    }

    /* Reset the group sizes */
    int *group_sizes = malloc(num_localpart * sizeof(int));
    for (long int i = 0; i < num_localpart; i++) {
        group_sizes[i] = 0;
    }

    /* Determine group sizes by counting particles with the same root */
    for (long int i = 0; i < num_localpart + receive_from_left; i++) {
        /* Skip foreign singletons */
        if (i >= num_localpart && fof_parts[i].root == fof_parts[i].global_offset) continue;
        /* Skip duplicate domestic particles */
        if (i >= num_localpart && (fof_parts[i].x[0] * int_to_rank_fac) == rank) continue;

        fof_parts[i].root = find_root(fof_parts, &fof_parts[i]);
        group_sizes[fof_parts[i].root]++;
    }

    /* Count the number of structures by tracing the ultimate roots */
    long int num_structures = 0;
    for (long int i = 0; i < num_localpart; i++) {
        /* Skip foreign singletons */
        if (i >= num_localpart && fof_parts[i].root == fof_parts[i].global_offset) continue;
        /* Skip duplicate domestic particles */
        if (i >= num_localpart && (fof_parts[i].x[0] * int_to_rank_fac) == rank) continue;

        if (fof_parts[i].local_offset == fof_parts[i].root && group_sizes[i] >= halo_min_npart) {
            num_structures++;
        }
    }

    /* Communicate the halo counts across all ranks */
    long long int *halos_per_rank = malloc(MPI_Rank_Count * sizeof(long int));
    long long int *halo_rank_offsets = malloc(MPI_Rank_Count * sizeof(long int));

    MPI_Allgather(&num_structures, 1, MPI_LONG_LONG, halos_per_rank,
                  1, MPI_LONG_LONG, MPI_COMM_WORLD);

    halo_rank_offsets[0] = 0;
    for (int i = 1; i < MPI_Rank_Count; i++) {
        halo_rank_offsets[i] = halo_rank_offsets[i - 1] + halos_per_rank[i - 1];
    }

    const long long int total_halo_num = halo_rank_offsets[MPI_Rank_Count - 1] + halos_per_rank[MPI_Rank_Count - 1];

    message(rank, "Found %ld structures in total.\n", total_halo_num);

    /* Allocate memory for the local halo catalogue */
    struct halo_properties *halos = malloc(num_structures * sizeof(struct halo_properties));
    bzero(halos, num_structures * sizeof(struct halo_properties));

    /* Assign root particles to halos */
    long int halo_count = 0;
    int *halo_ids = malloc(num_localpart * sizeof(int));
    for (long int i = 0; i < num_localpart; i++) {
        /* Skip foreign singletons */
        if (i >= num_localpart && fof_parts[i].root == fof_parts[i].global_offset) continue;
        /* Skip duplicate domestic particles */
        if (i >= num_localpart && (fof_parts[i].x[0] * int_to_rank_fac) == rank) continue;

        if (fof_parts[i].local_offset == fof_parts[i].root && group_sizes[i] >= halo_min_npart) {
            halo_ids[i] = halo_count + halo_rank_offsets[rank];
            halos[halo_count].global_id = halo_count + halo_rank_offsets[rank];
            halo_count++;
        } else {
            halo_ids[i] = -1;
        }
    }

    /* We are done with the group sizes */
    free(group_sizes);

    /* Accumulate the halo properties */
    for (long int i = 0; i < num_localpart + receive_from_left; i++) {
        /* Skip foreign singletons */
        if (i >= num_localpart && fof_parts[i].root == fof_parts[i].global_offset) continue;
        /* Skip duplicate domestic particles */
        if (i >= num_localpart && (fof_parts[i].x[0] * int_to_rank_fac) == rank) continue;

        long int h = halo_ids[fof_parts[i].root] - halo_rank_offsets[rank];
        if (h >= 0) {
#ifdef WITH_MASSES
            /* TODO: decide what to do about the masses */
            // double mass = fof_parts[i].m;
            double mass = 1.0;
#else
            double mass = 1.0;
#endif

            /* Friends-of-friends mass */
            halos[h].mass_fof += mass;
            /* Centre of mass */
            halos[h].x_com[0] += (int_to_pos_fac * fof_parts[i].x[0]) * mass;
            halos[h].x_com[1] += (int_to_pos_fac * fof_parts[i].x[1]) * mass;
            halos[h].x_com[2] += (int_to_pos_fac * fof_parts[i].x[2]) * mass;
            /* TODO: decide what to do about the velocities */
            /* Centre of mass velocity (convert to peculiar velocity) */
            // halos[h].v_com[0] += fof_parts[i].v[0] * mass / a_scale_factor;
            // halos[h].v_com[1] += fof_parts[i].v[1] * mass / a_scale_factor;
            // halos[h].v_com[2] += fof_parts[i].v[2] * mass / a_scale_factor;
            /* Total particle number */
            halos[h].npart++;
        }
    }

    /* We are done with the halo ids */
    free(halo_ids);

    /* Divide by the mass for the centre of mass properties */
    for (long int i = 0; i < num_structures; i++) {
        double halo_mass = halos[i].mass_fof;
        if (halo_mass > 0) {
            halos[i].x_com[0] /= halo_mass;
            halos[i].x_com[1] /= halo_mass;
            halos[i].x_com[2] /= halo_mass;
            // halos[i].v_com[0] /= halo_mass;
            // halos[i].v_com[1] /= halo_mass;
            // halos[i].v_com[2] /= halo_mass;
            // halos[i].rank_mean /= halos[i].npart;
        }
    }

    timer_stop(rank, &fof_timer, "Computing halo properties took ");

    /* Print the halo properties to a file */
    /* TODO: replace by HDF5 output */
    char fname[50];
    sprintf(fname, "halos_%04d_%03d.txt", output_num, rank);
    FILE *f = fopen(fname, "w");

    fprintf(f, "# i mass npart x[0] x[1] x[2] v[0] v[1] v[2]\n");
    for (long int i = 0; i < num_structures; i++) {
        fprintf(f, "%ld %g %d %g %g %g %g %g %g\n", halos[i].global_id, halos[i].mass_fof, halos[i].npart, halos[i].x_com[0], halos[i].x_com[1], halos[i].x_com[2], halos[i].v_com[0], halos[i].v_com[1], halos[i].v_com[2]);
    }

    /* Close the file */
    fclose(f);

    timer_stop(rank, &fof_timer, "Writing halo property files took ");

    /* Free all memory */
    free(fof_parts);
    free(cell_counts);
    free(cell_offsets);
    free(cell_list);
    free(parts_per_rank);
    free(rank_offsets);

    return 0;
}
