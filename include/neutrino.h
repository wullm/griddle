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

#ifndef NEUTRINO_H
#define NEUTRINO_H

#include "units.h"
#include "particle.h"
#include "fermi_dirac.h"

/* Neutrino delta-f weight (2010.07321) */
static inline void neutrino_weight(const FloatVelType v[3],
                                   const struct particle *p,
                                   const struct cosmology *cosmo,
                                   const double neutrino_qfac, double *q_out,
                                   double *w_out) {

#ifdef WITH_NEUTRINOS
    if (match_particle_type(p, neutrino_type, 0)) {

#if defined(WITH_PARTICLE_SEEDS)
        long int seed = p->seed;
#elif defined(WITH_PARTICLE_IDS)
        long int seed = p->id;
#else
        printf("Error: Not compiled with particle seeds or ids.\n");
        exit(1);
#endif

        double m_eV = cosmo->M_nu[(int)seed % cosmo->N_nu];
        double v2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
        double q = sqrt(v2) * neutrino_qfac * m_eV;
        double qi = neutrino_seed_to_fermi_dirac(seed);
        double f = fermi_dirac_density(q);
        double fi = fermi_dirac_density(qi);

        *q_out = q;
        *w_out = 1.0 - f / fi;
    } else {
        *q_out = 0.;
        *w_out = 1.;
    }
#else
    printf("Error: delta-f weight function called, but not compiled with neutrinos.\n");
    exit(1);
#endif
}

#endif
