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

#include "mass_deposit.h"

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
    char DoNewtonianBackscaling;

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
    double ScaleFactorTarget;
    int DerivativeOrder;

    /* Neutrino parameters */
    char NeutrinoPreIntegration;
    double NeutrinoScaleFactorEarly;
    double NeutrinoScaleFactorEarlyStep;

    /* Output time strings */
    char *SnapshotTimesString;
    char *PowerSpectrumTimesString;
    char *HaloFindingTimesString;

    /* Snapshot parameters */
    char *SnapshotBaseName;

    /* Halo finding parameters */
    char *CatalogueBaseName;
    char *SnipBaseName;
    char DoSphericalOverdensities;
    double LinkingLength;
    int MinHaloParticleNum;
    double HaloFindCellNumber;
    double SphericalOverdensityThreshold;
    double SphericalOverdensityMinLookRadius;
    double SphericalOverdensityMaxLookRadius;
    int FOFBufferSize;
    int CatalogueMassDScaleCompression;
    int CatalogueRadiusDScaleCompression;

    /* Shrinking sphere parameters */
    double ShrinkingSphereInitialRadius;
    double ShrinkingSphereRadiusFactorCoarse;
    double ShrinkingSphereRadiusFactor;
    double ShrinkingSphereMassFraction;
    int ShrinkingSphereMinParticleNum;

    /* Snipshot parameters */
    char ExportSnipshots;
    double SnipshotReduceFactor;
    int SnipshotMinParticleNum;
    int SnipshotPositionDScaleCompression;
    int SnipshotVelocityDScaleCompression;
    int SnipshotZipCompressionLevel;

    /* Power spectrum (for on-the-fly analysis) parameters */
    long long PowerSpectrumGridSize;
    int PowerSpectrumBins;
    int PositionDependentSplits;
    char *PowerSpectrumTypes;
};

void begin_used_parameter_file(const char *used_parameter_fname,
                               const char *fname, FILE **used_parameters_f,
                               int rank);
void begin_section(const char *section, FILE *used_parameters_f, int rank);
long read_long(const char *section, const char *key, long default_value,
               const char *fname, FILE *used_parameters_f, int rank);
double read_double(const char *section, const char *key, double default_value,
                   const char *fname, FILE *used_parameters_f, int rank);
int read_string(const char *section, const char *key, const char *default_value,
                char *buffer, int buffer_size, const char *fname,
                FILE *used_parameters_f, int rank);

int readParams(struct params *parser, const char *fname);
int cleanParams(struct params *parser);
int parseArrayString(char *string, double **array, int *length);
int parseCharArrayString(char *string, char ***array, int *length);
int parseOutputList(char *string, double **output_list, int *num_outputs,
                    double a_begin, double a_end);
int parseGridTypeList(char *string, enum grid_type **type_list, int *num_types);

#endif
