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

// /* Computing neutrino particle weights using kicked velocities, without
//  * actually changing the velocities to prevent round-off errors and maintain
//  * reversibility */
// int kick_weights_only(struct particle *parts, long int local_partnum,
//                       double a, double kick_dtau, double neutrino_qfac,
//                       const struct cosmology *cosmo,
//                       const struct physical_consts *pcs) {
// 
// #if defined(WITH_PARTTYPE) && defined(WITH_PARTICLE_IDS) && defined(WITH_ACCELERATIONS)
//     /* Drift the particles to the correct time */
//     for (long long i = 0; i < local_partnum; i++) {
//         struct particle *p = &parts[i];
// 
//         if (compare_particle_type(p, neutrino_type, 0)) {
//             /* Compute the kicked velocity (without changing p->v) */
//             FloatVelType v[3] = {p->v[0] + p->a[0] * kick_dtau,
//                                  p->v[1] + p->a[1] * kick_dtau,
//                                  p->v[2] + p->a[2] * kick_dtau};
// 
//             /* Update the particle weights */
//             double m_eV = cosmo->M_nu[(int)p->id % cosmo->N_nu];
//             double v2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
//             double q = sqrt(v2) * neutrino_qfac * m_eV;
//             double qi = neutrino_seed_to_fermi_dirac(p->id);
//             double f = fermi_dirac_density(q);
//             double fi = fermi_dirac_density(qi);
// 
//             p->w = 1.0 - f / fi;
//         }
//     }
// #endif
//     return 0;
// }