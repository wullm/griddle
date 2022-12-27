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

#ifndef PARAMS_H
#define PARAMS_H

#define DEFAULT_STRING_LENGTH 150

struct params {
    /* MPI rank (generated automatically) */
    int rank;

    /* Random parameters */
    long int Seed;
    char FixedModes;
    char InvertedModes;

    /* Initial conditions parameters */
    char GenerateICs;
    char *InitialConditionsFile;

    /* Perturbation file parameters */
    char *TransferFunctionsFile;

    /* Simulation parameters */
    long long PartGridSize;
    long long MeshGridSize;
    long long NeutrinosPerDim;
    long long ForeignBufferSize;
    double BoxLength;
    double ScaleFactorBegin;
    double ScaleFactorEnd;
    double ScaleFactorStep;
    int DerivativeOrder;

    /* Snapshot parameters */
    char *SnapshotTimesString;
    char *SnapshotBaseName;

    /* Halo finding parameters */
    char DoHaloFindingWithSnapshots;
    char DoSphericalOverdensities;
    double LinkingLength;
    int MinHaloParticleNum;
    char *CatalogueBaseName;
    char *SnipBaseName;

};

int readParams(struct params *parser, const char *fname);
int cleanParams(struct params *parser);
int parseArrayString(char *string, double **array, int *length);

#endif
