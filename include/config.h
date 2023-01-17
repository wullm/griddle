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

#ifndef CONFIG_H
#define CONFIG_H

/* Grid and Fourier transform options */
#define SINGLE_PRECISION_FFTW           /* Use single or double precision for the grids  */
#define MEASURED_FFTW_PLANS             /* Use measured FFTW plans */
#define USE_IN_PLACE_FFTS               /* Use in-place Fourier transforms */

/* Particle type options */
#define SINGLE_PRECISION_IDS            /* Use single or double precision particle IDs */
#define SINGLE_PRECISION_POSITIONS      /* Use single or double precision positions */
#define SINGLE_PRECISION_VELOCITIES     /* Use single or double precision velocities */

#define WITH_PARTICLE_SEEDS             /* Store random seeds for neutrino Fermi-Dirac momenta */
// #define WITH_ACCELERATIONS              /* Store particle accelerations for exact velocity outputs */
// #define WITH_MASSES                     /* Store individual particle masses */
// #define WITH_PARTICLE_IDS               /* Store particle IDs */
// #define WITH_PARTTYPE                /* Store particle type (CDM, neutrino) */

/* Other options */
#define WITH_NEUTRINOS                  /* Enable special treatment of neutrino particles */

#endif
