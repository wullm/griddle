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

#ifndef RELATIVITY_H
#define RELATIVITY_H

#include "units.h"
#include "particle.h"

/* Relativistic equations of motion (2207.14256) */
static inline double relativistic_drift(const FloatVelType v[3],
                                        const particle_data *p,
                                        const struct physical_consts *pcs,
                                        double a) {
#ifdef WITH_PARTTYPE
    if (p->type == 6) {
        FloatVelType v2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
        FloatVelType ac = a * pcs->SpeedOfLight;
        FloatVelType ac2 = ac * ac;
        return ac / sqrt(ac2 + v2);
    } else {
        return 1.0;
    }
#else
    return 1.0;
#endif
}

#endif
