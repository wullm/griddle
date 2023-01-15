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
#include <string.h>
#include <assert.h>
#include <math.h>
#include "../include/units.h"
#include "../include/params.h"

/* The .ini parser library is minIni */
#include "../parser/minIni.h"

int readUnits(struct units *us, const char *fname) {
    /* Get the dimensions of the cluster */
    int rank, MPI_Rank_Count;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &MPI_Rank_Count);

    /* Create a formatted file with the used parameters */
    FILE *f = NULL;
    begin_used_parameter_file("used_units.ini", fname, &f, rank);

    /* Internal units */
    begin_section("Units", f, rank);
    us->UnitLengthMetres = read_double("Units", "UnitLengthMetres", 1.0, fname, f, rank);
    us->UnitTimeSeconds = read_double("Units", "UnitTimeSeconds", 1.0, fname, f, rank);
    us->UnitMassKilogram = read_double("Units", "UnitMassKilogram", 1.0, fname, f, rank);
    us->UnitTemperatureKelvin = read_double("Units", "UnitTemperatureKelvin", 1.0, fname, f, rank);
    us->UnitCurrentAmpere = read_double("Units", "UnitCurrentAmpere", 1.0, fname, f, rank);

    /* Close the used parameters file */
    if (rank == 0) {
        fclose(f);
    }

    return 0;
}


int set_physical_constants(struct units *us, struct physical_consts *pcs) {
    /* Some physical constants */
    pcs->SpeedOfLight = SPEED_OF_LIGHT_METRES_SECONDS * us->UnitTimeSeconds
                        / us->UnitLengthMetres;
    pcs->GravityG = GRAVITY_G_SI_UNITS * us->UnitTimeSeconds * us->UnitTimeSeconds
                    / us->UnitLengthMetres / us->UnitLengthMetres / us->UnitLengthMetres
                    * us->UnitMassKilogram; // m^3 / kg / s^2 to internal
    pcs->hPlanck = PLANCK_CONST_SI_UNITS / us->UnitMassKilogram / us->UnitLengthMetres
                    / us->UnitLengthMetres * us->UnitTimeSeconds; //J*s = kg*m^2/s
    pcs->kBoltzmann = BOLTZMANN_CONST_SI_UNITS / us->UnitMassKilogram / us->UnitLengthMetres
                    / us->UnitLengthMetres * us->UnitTimeSeconds * us->UnitTimeSeconds
                    * us->UnitTemperatureKelvin; //J/K = kg*m^2/s^2/K
    pcs->ElectronVolt = ELECTRONVOLT_SI_UNITS / us->UnitMassKilogram / us->UnitLengthMetres
                    / us->UnitLengthMetres * us->UnitTimeSeconds
                    * us->UnitTimeSeconds; // J = kg*m^2/s^2
    pcs->SoundSpeedNeutrinos = NEUTRINO_SOUND_SPEED_1EV_SI_UNITS * us->UnitTimeSeconds
                        / us->UnitLengthMetres;
    return 0;
}
