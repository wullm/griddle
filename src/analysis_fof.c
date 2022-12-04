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
#include "../include/analysis_so.h"
#include "../include/message.h"

#define DEBUG_CHECKS

MPI_Datatype fof_type;

/* Should two particles be linked? */
static inline int should_link(const IntPosType ax[3], const IntPosType bx[3],
                              double int_to_pos_fac, double linking_length_2) {

    const double r_2 = int_to_phys_dist2(ax, bx, int_to_pos_fac);
    return r_2 <= linking_length_2;
}


/* Find the root of the set of a given particle */
long int find_root(struct fof_part_data *fof_parts, struct fof_part_data *part) {

#ifdef DEBUG_CHECKS
    assert(part->root >= 0);
#endif

    if (part->local_offset == part->root)
        return part->local_offset;
    part->root = find_root(fof_parts, &fof_parts[part->root]);
    return part->root;
}

/* Perform the union operation on the sets containing two given particles */
void union_roots(struct fof_part_data *fof_parts, struct fof_part_data *a, struct fof_part_data *b) {

#ifdef DEBUG_CHECKS
    assert(a->root >= 0);
    assert(b->root >= 0);
#endif

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

/* Receive FOF particle data */
void receive_fof_parts(struct fof_part_data *dest, int *num_received,
                       int from_rank, long long int num_localpart,
                       long long int max_partnum) {

    /* Prepare to receive particles */
    MPI_Status status;
    MPI_Probe(from_rank, 0, MPI_COMM_WORLD, &status);
    MPI_Get_count(&status, fof_type, num_received);

    /* Check that we have enough memory */
    if (num_localpart + *num_received > max_partnum) {
        printf("Not enough memory to exchange FOF data (%lld < %lld).\n", max_partnum, num_localpart + *num_received);
        exit(1);
    }

    /* Receive the particle data */
    MPI_Recv(dest, *num_received, fof_type, from_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

/* Compute the distance from a particle to a given edge:
* type = 0: left edge of the rank
* type = 1: right edge of the domain */
double edge_distance(IntPosType x, double boxlen, double int_to_rank_fac,
                     double rank_to_pos_fac, double int_to_pos_fac, int type) {

    if (type == 0) {
        /* Distance from left edge of rank */
        double rank_float = x * int_to_rank_fac;
        double dx = rank_float - ((int) rank_float);
        return dx * rank_to_pos_fac;
    } else {
        /* Distance from right edge of domain */
        double pos = x * int_to_pos_fac;
        return boxlen - pos;
    }
}

/* Copy particles within a linking length from an edge. There are two types:
 * type = 0: left edge of the rank
 * type = 1: right edge of the domain */
void copy_edge_parts(struct fof_part_data **dest, struct fof_part_data *fof_parts,
                     int *num_copied, long long int num_localpart, int type,
                     double int_to_rank_fac, double rank_to_pos_fac,
                     double int_to_pos_fac, double boxlen, double linking_length) {

    /* Count the number of particles within one linking length from the left edge */
    int count_near_edge = 0;
    for (long int i = 0; i < num_localpart; i++) {
        double dx = edge_distance(fof_parts[i].x[0], boxlen, int_to_rank_fac, rank_to_pos_fac, int_to_pos_fac, type);
        if (dx < linking_length) {
            count_near_edge++;
        }
    }

    /* Allocate memory for the edge particles */
    *dest = malloc(count_near_edge * sizeof(struct fof_part_data));
    *num_copied = count_near_edge;

    /* Fish out particles within one linking length from the left edge */
    int copy_counter = 0;
    for (long int i = 0; i < num_localpart; i++) {
        double dx = edge_distance(fof_parts[i].x[0], boxlen, int_to_rank_fac, rank_to_pos_fac, int_to_pos_fac, type);
        if (dx < linking_length) {
            memcpy(*dest + copy_counter, fof_parts + i, sizeof(struct fof_part_data));
            copy_counter++;
        }
    }
}

/* TODO, kick and drift particles to the right time */
int analysis_fof(struct particle *parts, double boxlen, long int Np,
                 long long int Ng, long long int num_localpart,
                 long long int max_partnum, double linking_length,
                 int halo_min_npart, int output_num, double a_scale_factor,
                 const struct units *us, const struct physical_consts *pcs,
                 const struct cosmology *cosmo) {

    /* Get the dimensions of the cluster */
    int rank, MPI_Rank_Count;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &MPI_Rank_Count);

    /* The MPI ranks are placed along a periodic ring */
    int rank_left = (rank == 0) ? MPI_Rank_Count - 1 : rank - 1;
    int rank_right = (rank + 1) % MPI_Rank_Count;

    /* Data type for MPI communication of particles */
    fof_type = mpi_fof_part_type();

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

    /* The cells must be larger than the linking length */
    if (boxlen / N_cells <= 2 * linking_length) {
        printf("The cells are smaller than the linking length, which is bad.\n");
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

    /* The total number of particles to be received */
    int receive_foreign_count = 0;

    /* When running with multiple ranks, we need to communicate edge particles */
    if (MPI_Rank_Count > 1) {

        /* We aim to communicate towards ranks with the lower number, which is
         * usually to the left, except for the last rank. Hence, three cases:
         *
         * 1) Rank 0             receives particles from both left and right
         * 2) Ranks 1 ... N-2    receive particles from right and send to left
         * 3) Rank N-1           sends particles to left and right
         *
         */
        if (rank == 0) {
            int receive_from_right, receive_from_left;

            /* Receive particle data from the right */
            receive_fof_parts(fof_parts + num_localpart, &receive_from_right,
                              rank_right, num_localpart, max_partnum);

            /* Receive particle data from the left */
            receive_fof_parts(fof_parts + num_localpart + receive_from_right,
                              &receive_from_left, rank_left, num_localpart,
                              max_partnum);

            receive_foreign_count += receive_from_right;
            receive_foreign_count += receive_from_left;
        } else if (rank < MPI_Rank_Count - 1) {
            /* Receive particles from the right */
            receive_fof_parts(fof_parts + num_localpart, &receive_foreign_count,
                              rank_right, num_localpart, max_partnum);

            struct fof_part_data *edge_parts;
            int count_near_edge;

            /* Fish out particles within one linking length from the left edge */
            copy_edge_parts(&edge_parts, fof_parts, &count_near_edge,
                            num_localpart, /* left */ 0, int_to_rank_fac,
                            rank_to_pos_fac, int_to_pos_fac, boxlen,
                            linking_length);

            /* Communicate the edge particles to the left neighbour */
            MPI_Send(edge_parts, count_near_edge, fof_type, rank_left, 0, MPI_COMM_WORLD);

            /* Release the delivered particles */
            free(edge_parts);
        } else {
            struct fof_part_data *edge_parts;
            int count_near_edge;

            /* Fish out particles within one linking length from the left edge */
            copy_edge_parts(&edge_parts, fof_parts, &count_near_edge,
                            num_localpart, /* left */ 0, int_to_rank_fac,
                            rank_to_pos_fac, int_to_pos_fac, boxlen,
                            linking_length);

            /* Communicate the edge particles to the left neighbour */
            MPI_Send(edge_parts, count_near_edge, fof_type, rank_left, 0, MPI_COMM_WORLD);

            /* Release the delivered particles */
            free(edge_parts);

            /* Fish out particles within one linking length from the right edge */
            copy_edge_parts(&edge_parts, fof_parts, &count_near_edge,
                            num_localpart, /* right */ 1, int_to_rank_fac,
                            rank_to_pos_fac, int_to_pos_fac, boxlen,
                            linking_length);

            /* Communicate these edge particles to the right neighbour */
            MPI_Send(edge_parts, count_near_edge, fof_type, rank_right, 0, MPI_COMM_WORLD);

            /* Release the delivered particles */
            free(edge_parts);
        }

        timer_stop(rank, &fof_timer, "The first communication took ");
    }

    /* Set the local offsets for the local and foreign particles */
    for (long long i = 0; i < num_localpart + receive_foreign_count; i++) {
        fof_parts[i].local_offset = i;
        fof_parts[i].root = i;
    }

    /* Now create a particle-cell correspondence for sorting */
    struct fof_cell_list *cell_list = malloc((num_localpart + receive_foreign_count) * sizeof(struct fof_cell_list));
    for (long long i = 0; i < num_localpart + receive_foreign_count; i++) {
        cell_list[i].cell = which_cell(fof_parts[i].x, int_to_cell_fac, N_cells);
        cell_list[i].offset = i;
    }

    /* Sort particles by cell */
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

    /* Now link particles within and between neighbouring cells */
    for (int i = 0; i < N_cells; i++) {
        for (int j = 0; j < N_cells; j++) {
            for (int k = 0; k < N_cells; k++) {

                long int offset = cell_offsets[row_major_cell(i, j, k, N_cells)];
                long int count = cell_counts[row_major_cell(i, j, k, N_cells)];

                /* Loop over the 27 neighbour cells (including itself) */
                for (int u = -1; u <= 1; u++) {
                    for (int v = -1; v <= 1; v++) {
                        for (int w = -1; w <= 1; w++) {
                            int i1 = i + u;
                            int j1 = j + v;
                            int k1 = k + w;

                            /* Account for periodic boundary conditions */
                            if (i1 >= N_cells) i1 -= N_cells;
                            if (j1 >= N_cells) j1 -= N_cells;
                            if (k1 >= N_cells) k1 -= N_cells;
                            if (i1 < 0) i1 += N_cells;
                            if (j1 < 0) j1 += N_cells;
                            if (k1 < 0) k1 += N_cells;

                            long int offset1 = cell_offsets[row_major_cell(i1, j1, k1, N_cells)];
                            long int count1 = cell_counts[row_major_cell(i1, j1, k1, N_cells)];

                            /* Link cells */
                            total_links += link_cells(fof_parts, cell_list, offset, offset1, count, count1, int_to_pos_fac, linking_length_2);
                        }
                    }
                }
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

    /* Now we can replace the roots by the global roots */
    for (long int i = 0; i < num_localpart + receive_foreign_count; i++) {
        fof_parts[i].root = global_roots[i];
    }

    /* Free the global root array */
    free(global_roots);

    /* The total number of particles to be received from the left */
    int receive_from_left = 0;

    /* When running with multiple ranks, we need to communicate edge particles */
    if (MPI_Rank_Count > 1) {

        /* Now all ranks (except the last rank) may have particles with a root
         * particle that belongs to a rank with a higher number. Two cases:
         *
         * Rank 0          sends particles to the right
         * Rank 1...N-1    receive from the left and send to the right
         *
         * This will automatically account for structures that link across
         * multiple ranks and deliver particles from rank 0 to rank N-1,
         * although not in the most efficient way.
         */
        if (rank == 0) {
            /* Determine the number of particles with foreign roots */
            int num_foreign_rooted = 0;
            for (long int i = 0; i < num_localpart + receive_foreign_count; i++) {
                if (fof_parts[i].root < rank_offset || fof_parts[i].root >= rank_offset + num_localpart) {
                    num_foreign_rooted++;
                }
            }

            /* Now, fish out all particles with foreign roots */
            long int copy_counter = 0;
            struct fof_part_data *foreign_root_parts = malloc(num_foreign_rooted * sizeof(struct fof_part_data));
            for (long int i = 0; i < num_localpart + receive_foreign_count; i++) {
                if (fof_parts[i].root < rank_offset || fof_parts[i].root >= rank_offset + num_localpart) {
                    memcpy(foreign_root_parts + copy_counter, fof_parts + i, sizeof(struct fof_part_data));
                    copy_counter++;
                }
            }

            /* Disable the remaining copies of the foreign rooted particles on
             * this rank that now live on another rank */
            for (long int i = 0; i < num_localpart + receive_foreign_count; i++) {
                if (fof_parts[i].root < rank_offset || fof_parts[i].root >= rank_offset + num_localpart) {
                    fof_parts[i].root = -1;
                }
            }

            /* Communicate the foreign particles to the right neighbour */
            MPI_Send(foreign_root_parts, num_foreign_rooted, fof_type, rank_right, 0, MPI_COMM_WORLD);

            /* Release the delivered particles */
            free(foreign_root_parts);
        } else {
            /* Receive particles from the left */
            receive_fof_parts(fof_parts + num_localpart + receive_foreign_count,
                              &receive_from_left, rank_left, num_localpart,
                              max_partnum);

            /* After receiving (possibly still foreign rooted) particles,
             * determine the number of particles with foreign roots */
            int num_foreign_rooted = 0;
            for (long int i = 0; i < num_localpart + receive_foreign_count + receive_from_left; i++) {
                /* Skip disabled particles */
                if (fof_parts[i].root == -1) continue;

                if (fof_parts[i].root < rank_offset || fof_parts[i].root >= rank_offset + num_localpart) {
                    num_foreign_rooted++;
                }
            }

            /* Now, fish out particles with foreign roots */
            long int copy_counter = 0;
            struct fof_part_data *foreign_root_parts = malloc(num_foreign_rooted * sizeof(struct fof_part_data));
            for (long int i = 0; i < num_localpart + receive_foreign_count + receive_from_left; i++) {
                /* Skip disabled particles */
                if (fof_parts[i].root == -1) continue;

                if (fof_parts[i].root < rank_offset || fof_parts[i].root >= rank_offset + num_localpart) {
                    memcpy(foreign_root_parts + copy_counter, fof_parts + i, sizeof(struct fof_part_data));
                    copy_counter++;
                }
            }

            /* Disable the remaining copies of the foreign rooted particles on
             * this rank that now live on another rank */
            for (long int i = 0; i < num_localpart + receive_foreign_count + receive_from_left; i++) {
                if (fof_parts[i].root < rank_offset || fof_parts[i].root >= rank_offset + num_localpart) {
                    fof_parts[i].root = -1;
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

#ifdef DEBUG_CHECKS
    /* Check that we have no foreign rooted particles */
    for (long int i = 0; i < num_localpart + receive_foreign_count + receive_from_left; i++) {
        /* Skip disabled particles */
        if (fof_parts[i].root == -1) continue;

        assert(fof_parts[i].root >= rank_offset && fof_parts[i].root < rank_offset + num_localpart);
    }
#endif

    /* Now we have no foreign rooted particles. This means that all particles
       that are linked to something are located on the "home rank" of the root
       particle of that group. Now we can attach the trees. */

    /* Turn the global_roots back into roots. This is safe because no particle
     * has a foreign root. Additionally, update the local offsets.  */
    for (long long i = 0; i < num_localpart + receive_foreign_count + receive_from_left; i++) {
        /* Skip disabled particles */
        if (fof_parts[i].root == -1) continue; // don't remove

        fof_parts[i].root -= rank_offset;
        fof_parts[i].local_offset = i;
    }

    /* Attach the trees of the received copies of local particles to the trees
     * of their local copies */
    for (long int i = num_localpart; i < num_localpart + receive_foreign_count + receive_from_left; i++) {
        /* Skip disabled particles */
        if (fof_parts[i].root == -1) continue;

        /* Is this a copy of a local particle? */
        if (fof_parts[i].global_offset >= rank_offset && fof_parts[i].global_offset < rank_offset + num_localpart) {
            /* Attach the trees */
            long int local_copy = fof_parts[i].global_offset - rank_offset;
            union_roots(fof_parts, &fof_parts[i], &fof_parts[local_copy]);

            /* Now disable the particle */
            fof_parts[i].root = -1;
        }
    }

    /* Attach received particles to the local tree using the proper local offsets */
    for (long long i = num_localpart; i < num_localpart + receive_foreign_count + receive_from_left; i++) {
        /* Skip disabled particles */
        if (fof_parts[i].root == -1) continue;

        union_roots(fof_parts, &fof_parts[i], &fof_parts[fof_parts[i].root]);
    }

    /* Allocate memory for the group sizes */
    int *group_sizes = calloc(num_localpart, sizeof(int));

    /* Determine group sizes by counting particles with the same root */
    for (long int i = 0; i < num_localpart + receive_foreign_count + receive_from_left; i++) {
        /* Skip disabled particles */
        if (fof_parts[i].root == -1) continue;

        fof_parts[i].root = find_root(fof_parts, &fof_parts[i]);
        group_sizes[fof_parts[i].root]++;
    }

    /* Count the number of structures by tracing the ultimate roots */
    long int num_structures = 0;
    for (long int i = 0; i < num_localpart; i++) {
        /* Skip disabled particles */
        if (fof_parts[i].root == -1) continue;

        if (fof_parts[i].local_offset == fof_parts[i].root && group_sizes[i] >= halo_min_npart) {
            num_structures++;
        }
    }

    /* Communicate the halo counts across all ranks */
    long long int *halos_per_rank = malloc(MPI_Rank_Count * sizeof(long int));
    long long int *halo_rank_offsets = malloc(MPI_Rank_Count * sizeof(long int));

    MPI_Allgather(&num_structures, 1, MPI_LONG_LONG, halos_per_rank, 1,
                  MPI_LONG_LONG, MPI_COMM_WORLD);

    halo_rank_offsets[0] = 0;
    for (int i = 1; i < MPI_Rank_Count; i++) {
        halo_rank_offsets[i] = halo_rank_offsets[i - 1] + halos_per_rank[i - 1];
    }

    const long long int total_halo_num = halo_rank_offsets[MPI_Rank_Count - 1] + halos_per_rank[MPI_Rank_Count - 1];

    message(rank, "Found %ld structures in total.\n", total_halo_num);

    /* Allocate memory for the local halo catalogue */
    struct fof_halo *halos = malloc(num_structures * sizeof(struct fof_halo));
    bzero(halos, num_structures * sizeof(struct fof_halo));

    /* Assign root particles to halos */
    long int halo_count = 0;
    int *halo_ids = malloc(num_localpart * sizeof(int));
    for (long int i = 0; i < num_localpart; i++) {
        /* Skip disabled particles */
        if (fof_parts[i].root == -1) continue;

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
    for (long int i = 0; i < num_localpart + receive_foreign_count + receive_from_left; i++) {
        /* Skip disabled particles */
        if (fof_parts[i].root == -1) continue;

        long int root = fof_parts[i].root;

        long int h = halo_ids[root] - halo_rank_offsets[rank];
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

            /* Compute the offset from the root particle */
            const IntPosType dx = fof_parts[i].x[0] - fof_parts[root].x[0];
            const IntPosType dy = fof_parts[i].x[1] - fof_parts[root].x[1];
            const IntPosType dz = fof_parts[i].x[2] - fof_parts[root].x[2];

            /* Enforce boundary conditions and convert to physical lengths */
            const double fx = (dx < -dx) ? dx * int_to_pos_fac : -((-dx) * int_to_pos_fac);
            const double fy = (dy < -dy) ? dy * int_to_pos_fac : -((-dy) * int_to_pos_fac);
            const double fz = (dz < -dz) ? dz * int_to_pos_fac : -((-dz) * int_to_pos_fac);

            /* Centre of mass (use offset from root for periodic boundary conditions) */
            halos[h].x_com[0] += (int_to_pos_fac * fof_parts[root].x[0] + fx) * mass;
            halos[h].x_com[1] += (int_to_pos_fac * fof_parts[root].x[1] + fy) * mass;
            halos[h].x_com[2] += (int_to_pos_fac * fof_parts[root].x[2] + fz) * mass;
            /* Total particle number */
            halos[h].npart++;
            /* The home rank of the halo */
            halos[h].rank = rank;
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
        }
    }

    /* Print the FOF properties to a file */
    /* TODO: replace by HDF5 output */
    char fname[50];
    sprintf(fname, "halos_FOF_%04d_%03d.txt", output_num, rank);
    FILE *f = fopen(fname, "w");

    fprintf(f, "# i M_FOF npart_FOF x[0] x[1] x[2]\n");
    for (long int i = 0; i < num_structures; i++) {
        fprintf(f, "%ld %g %d %g %g %g\n", halos[i].global_id, halos[i].mass_fof, halos[i].npart, halos[i].x_com[0], halos[i].x_com[1], halos[i].x_com[2]);
    }

    /* Close the file */
    fclose(f);

    /* Timer */
    timer_stop(rank, &fof_timer, "Writing FOF halo properties took ");

    message(rank, "\n");
    message(rank, "Proceeding with spherical overdensity calculations.\n");

    analysis_so(parts, halos, boxlen, Np, Ng, num_localpart, max_partnum,
                num_structures, output_num, a_scale_factor,us,pcs, cosmo);


    /* Free all memory */
    free(fof_parts);
    free(cell_counts);
    free(cell_offsets);
    free(cell_list);
    free(parts_per_rank);
    free(rank_offsets);

    return 0;
}
