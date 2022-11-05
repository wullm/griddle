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

#ifndef PARTICLE_H
#define PARTICLE_H

#include <stdint.h>

#define SINGLE_PRECISION_POSITIONS

#ifdef SINGLE_PRECISION_POSITIONS
#define POSITION_BITS 32
typedef uint32_t IntPosType;
#else
#define POSITION_BITS 64
typedef uint64_t IntPosType;
#endif

struct particle {
    /* Basic particle data */
    long long int id;
    char type;

    /* Position, velocity, mass */
    IntPosType x[3];
    double v[3];
    double m;

    /* Neutrino delta-f weight */
    double w;

    /* Most recent accelerations */
    double a[3];

    /* Communication data */
    int rank;
    int exchange_dir;
};

static inline int particleSort(const void *a, const void *b) {
    struct particle *pa = (struct particle*) a;
    struct particle *pb = (struct particle*) b;
    return pa->exchange_dir >= pb->exchange_dir;
}


static inline int particleTypeSort(const void *a, const void *b) {
    struct particle *pa = (struct particle*) a;
    struct particle *pb = (struct particle*) b;
    return pa->type >= pb->type;
}

#endif
