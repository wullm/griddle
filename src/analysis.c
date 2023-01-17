/*******************************************************************************
 * This file is part of Sedulus.
 * Copyright (c) 2023 Willem Elbers (whe@willemelbers.com)
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

#include <math.h>
#include <mpi.h>
#include "../include/analysis.h"
#include "../include/relativity.h"

IntPosType position_checksum(const struct particle *parts, long int local_partnum) {
    /* Get the dimensions of the cluster */
    int rank, MPI_Rank_Count;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &MPI_Rank_Count);
    
    /* Determine position checksum */
    IntPosType checksum_local = 0;

    for (long long i = 0; i < local_partnum; i++) {
        const struct particle *p = &parts[i];
        checksum_local += p->x[0];
        checksum_local += p->x[1];
        checksum_local += p->x[2];
    }

    IntPosType checksum_global = 0;
    MPI_Reduce(&checksum_local, &checksum_global, 1,
               MPI_UNSIGNED, MPI_SUM, /* root = */ 0, MPI_COMM_WORLD);
               
    return checksum_global;
}

int drift_particles(struct particle *parts, long int local_partnum,
                    double a, double drift_dtau, double pos_to_int_fac,
                    const struct physical_consts *pcs) {

    /* Drift the particles to the correct time */
    for (long long i = 0; i < local_partnum; i++) {
        struct particle *p = &parts[i];

        /* Relativistic drift correction */
        const double rel_drift = relativistic_drift(p->v, p, pcs, a);

        /* Execute drift */
        p->x[0] += (IntPosType) (p->v[0] * drift_dtau * rel_drift * pos_to_int_fac);
        p->x[1] += (IntPosType) (p->v[1] * drift_dtau * rel_drift * pos_to_int_fac);
        p->x[2] += (IntPosType) (p->v[2] * drift_dtau * rel_drift * pos_to_int_fac);
    }                        
    return 0;                        
}