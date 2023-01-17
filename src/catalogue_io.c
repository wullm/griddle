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

/* Methods for input and output of halo catalogues */

#include <stdlib.h>
#include <string.h>
#include <hdf5.h>
#include <assert.h>
#include <math.h>
#include "../include/analysis_fof.h"
#include "../include/analysis_so.h"
#include "../include/catalogue_io.h"
#include "../include/git_version.h"

/* The current limit of for parallel HDF5 writes is 2GB */
#define HDF5_PARALLEL_LIMIT 2147000000LL
/* TODO: Not the default chunk size, decide what makes sense */
#define HDF5_CHUNK_SIZE 128LL

int writeCatalogueAttributes(const struct params *pars, const struct units *us,
                             double a, hid_t h_file) {

    /* Create the Header group */
    hid_t h_grp = H5Gcreate(h_file, "/Header", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    /* Retrieve the number of ranks */
    int MPI_Rank_Count;
    MPI_Comm_size(MPI_COMM_WORLD, &MPI_Rank_Count);

    /* Create dataspace for BoxSize attribute */
    const hsize_t arank = 1;
    const hsize_t adims_three[1] = {3}; //3D space
    hid_t h_aspace = H5Screate_simple(arank, adims_three, NULL);

    /* Create the BoxSize attribute and write the data */
    hid_t h_attr = H5Acreate1(h_grp, "BoxSize", H5T_NATIVE_DOUBLE, h_aspace, H5P_DEFAULT);
    double boxlen = pars->BoxLength;
    double boxsize[3] = {boxlen, boxlen, boxlen};
    H5Awrite(h_attr, H5T_NATIVE_DOUBLE, boxsize);
    H5Aclose(h_attr);

    /* Change dataspace dimensions to scalar value attributes */
    const hsize_t adims_single[1] = {1};
    H5Sset_extent_simple(h_aspace, arank, adims_single, NULL);

    /* Create the Dimension attribute and write the data */
    int dimension = 3;
    h_attr = H5Acreate1(h_grp, "Dimension", H5T_NATIVE_INT, h_aspace, H5P_DEFAULT);
    H5Awrite(h_attr, H5T_NATIVE_INT, &dimension);
    H5Aclose(h_attr);

    /* Create the Redshift attribute and write the data */
    double z_output = 1./a - 1;
    h_attr = H5Acreate1(h_grp, "Redshift", H5T_NATIVE_DOUBLE, h_aspace, H5P_DEFAULT);
    H5Awrite(h_attr, H5T_NATIVE_DOUBLE, &z_output);
    H5Aclose(h_attr);

    /* Store the linking length parameter */
    h_attr = H5Acreate1(h_grp, "LinkingLength", H5T_NATIVE_DOUBLE, h_aspace, H5P_DEFAULT);
    H5Awrite(h_attr, H5T_NATIVE_DOUBLE, &pars->LinkingLength);
    H5Aclose(h_attr);
    
    /* Store the actual dimensionful linking length parameter */
    double linking_length = pars->LinkingLength * pars->BoxLength / pars->PartGridSize;
    h_attr = H5Acreate1(h_grp, "LinkingLengthComoving", H5T_NATIVE_DOUBLE, h_aspace, H5P_DEFAULT);
    H5Awrite(h_attr, H5T_NATIVE_DOUBLE, &linking_length);
    H5Aclose(h_attr);
    
    /* Store the minimum particle number parameter */
    h_attr = H5Acreate1(h_grp, "MinHaloParticleNum", H5T_NATIVE_DOUBLE, h_aspace, H5P_DEFAULT);
    H5Awrite(h_attr, H5T_NATIVE_INT, &pars->MinHaloParticleNum);
    H5Aclose(h_attr);

    /* Variable-length string type */
    hid_t vlstrtype = H5Tcopy(H5T_C_S1);

    /* Write git versioning information to the snapshot */
    H5Tset_size(vlstrtype, strlen(GIT_BRANCH));
    h_attr = H5Acreate1(h_grp, "GitBranch", vlstrtype, h_aspace, H5P_DEFAULT);
    H5Awrite(h_attr, vlstrtype, GIT_BRANCH);
    H5Aclose(h_attr);

    H5Tset_size(vlstrtype, strlen(GIT_COMMIT));
    h_attr = H5Acreate1(h_grp, "GitCommit", vlstrtype, h_aspace, H5P_DEFAULT);
    H5Awrite(h_attr, vlstrtype, GIT_COMMIT);
    H5Aclose(h_attr);

    H5Tset_size(vlstrtype, strlen(GIT_DIRTY_HASH));
    h_attr = H5Acreate1(h_grp, "GitUnstagedChecksum", vlstrtype, h_aspace, H5P_DEFAULT);
    H5Awrite(h_attr, vlstrtype, GIT_DIRTY_HASH);
    H5Aclose(h_attr);

    H5Tset_size(vlstrtype, strlen(GIT_MESSAGE));
    h_attr = H5Acreate1(h_grp, "GitMessage", vlstrtype, h_aspace, H5P_DEFAULT);
    H5Awrite(h_attr, vlstrtype, GIT_MESSAGE);
    H5Aclose(h_attr);

    H5Tset_size(vlstrtype, strlen(GIT_DATE));
    h_attr = H5Acreate1(h_grp, "GitDate", vlstrtype, h_aspace, H5P_DEFAULT);
    H5Awrite(h_attr, vlstrtype, GIT_DATE);
    H5Aclose(h_attr);

    /* Close the attribute dataspace */
    H5Sclose(h_aspace);

    /* Close the Header group */
    H5Gclose(h_grp);

    /* Create the Cosmology group */
    h_grp = H5Gcreate(h_file, "/Cosmology", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    /* Create dataspace for scalar value attributes */
    h_aspace = H5Screate_simple(arank, adims_single, NULL);

    /* Create the Redshift attribute and write the data */
    h_attr = H5Acreate1(h_grp, "Redshift", H5T_NATIVE_DOUBLE, h_aspace, H5P_DEFAULT);
    H5Awrite(h_attr, H5T_NATIVE_DOUBLE, &z_output);
    H5Aclose(h_attr);

    /* Close the attribute dataspace */
    H5Sclose(h_aspace);

    /* Close the Cosmology group */
    H5Gclose(h_grp);

    /* Create the Units group */
    h_grp = H5Gcreate(h_file, "/Units", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    /* Create dataspace for scalar value attributes */
    h_aspace = H5Screate_simple(arank, adims_single, NULL);

    /* Determine the units used */
    double unit_mass_cgs = us->UnitMassKilogram * 1000;
    double unit_length_cgs = us->UnitLengthMetres * 100;
    double unit_time_cgs = us->UnitTimeSeconds;
    double unit_temperature_cgs = us->UnitTemperatureKelvin;
    double unit_current_cgs = us->UnitCurrentAmpere;

    /* Write the internal unit system */
    h_attr = H5Acreate1(h_grp, "Unit mass in cgs (U_M)", H5T_NATIVE_DOUBLE, h_aspace, H5P_DEFAULT);
    H5Awrite(h_attr, H5T_NATIVE_DOUBLE, &unit_mass_cgs);
    H5Aclose(h_attr);

    h_attr = H5Acreate1(h_grp, "Unit length in cgs (U_L)", H5T_NATIVE_DOUBLE, h_aspace, H5P_DEFAULT);
    H5Awrite(h_attr, H5T_NATIVE_DOUBLE, &unit_length_cgs);
    H5Aclose(h_attr);

    h_attr = H5Acreate1(h_grp, "Unit time in cgs (U_t)", H5T_NATIVE_DOUBLE, h_aspace, H5P_DEFAULT);
    H5Awrite(h_attr, H5T_NATIVE_DOUBLE, &unit_time_cgs);
    H5Aclose(h_attr);

    h_attr = H5Acreate1(h_grp, "Unit temperature in cgs (U_T)", H5T_NATIVE_DOUBLE, h_aspace, H5P_DEFAULT);
    H5Awrite(h_attr, H5T_NATIVE_DOUBLE, &unit_temperature_cgs);
    H5Aclose(h_attr);

    h_attr = H5Acreate1(h_grp, "Unit current in cgs (U_I)", H5T_NATIVE_DOUBLE, h_aspace, H5P_DEFAULT);
    H5Awrite(h_attr, H5T_NATIVE_DOUBLE, &unit_current_cgs);
    H5Aclose(h_attr);

    /* Close the attribute dataspace */
    H5Sclose(h_aspace);

    /* Close the Cosmology group */
    H5Gclose(h_grp);


    return 0;
}

int exportCatalogue(const struct params *pars, const struct units *us,
                    const struct physical_consts *pcs, int output_num, double a,
                    long int total_num_structures, long int local_num_structures,
                    struct fof_halo *fofs) {

    /* Get the dimensions of the cluster */
    int rank, MPI_Rank_Count;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &MPI_Rank_Count);

    /* Form the filename */
    char fname[DEFAULT_STRING_LENGTH];
    sprintf(fname, "%s_%04d.hdf5", pars->CatalogueBaseName, output_num);

    /* The HDF5 group name of each halo type */
    const int num_types = pars->DoSphericalOverdensities ? 2 : 1;
    char ExportNames[2][100];
    sprintf(ExportNames[0], "FOF");
    sprintf(ExportNames[1], "SO_%d_crit", (int) pars->SphericalOverdensityThreshold);

    if (rank == 0) {
        /* Create the output file */
        hid_t h_out_file = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

        /* Writing attributes into the Header & Cosmology groups */
        int err = writeCatalogueAttributes(pars, us, a, h_out_file);
        if (err > 0) exit(1);

        /* The particle group in the output file */
        hid_t h_grp;
        
        /* Datsets */
        hid_t h_data;
        
        /* For each halo type, prepare the group and data sets */
        for (int t = 0; t < num_types; t++) {
        
            /* Vector dataspace (e.g. positions, velocities) */
            const hsize_t vrank = 2;
            const hsize_t vdims[2] = {total_num_structures, 3};
            hid_t h_vspace = H5Screate_simple(vrank, vdims, NULL);
        
            /* Scalar dataspace (e.g. masses, ids) */
            const hsize_t srank = 1;
            const hsize_t sdims[1] = {total_num_structures};
            hid_t h_sspace = H5Screate_simple(srank, sdims, NULL);
        
            /* Set chunking for vectors and scalars */
            const hsize_t vchunk[2] = {HDF5_CHUNK_SIZE, 3};
            const hsize_t schunk[1] = {HDF5_CHUNK_SIZE};
            
            /* Create properties for scalars */
            hid_t h_prop_sca = H5Pcreate(H5P_DATASET_CREATE);
            H5Pset_chunk(h_prop_sca, srank, schunk);

            /* Create properties for the positions */
            hid_t h_prop_pos = H5Pcreate(H5P_DATASET_CREATE);
            H5Pset_chunk(h_prop_pos, vrank, vchunk);

            /* Create properties for the positions */
            hid_t h_prop_vel = H5Pcreate(H5P_DATASET_CREATE);
            H5Pset_chunk(h_prop_vel, vrank, vchunk);

            /* Create properties for the masses */
            hid_t h_prop_mass = H5Pcreate(H5P_DATASET_CREATE);
            H5Pset_chunk(h_prop_mass, srank, schunk);

            /* Create properties for the radii */
            hid_t h_prop_radius = H5Pcreate(H5P_DATASET_CREATE);
            H5Pset_chunk(h_prop_radius, srank, schunk);

            /* Set lossy filter (keep "digits" digits after the decimal point) */
            const int digits_pos = pars->SnipshotPositionDScaleCompression;
            const int digits_vel = pars->SnipshotVelocityDScaleCompression;
            const int digits_mass = pars->CatalogueMassDScaleCompression;
            const int digits_radius = pars->CatalogueRadiusDScaleCompression;
            if (digits_pos > 0)
                H5Pset_scaleoffset(h_prop_pos, H5Z_SO_FLOAT_DSCALE, digits_pos);
            if (digits_vel > 0)
                H5Pset_scaleoffset(h_prop_vel, H5Z_SO_FLOAT_DSCALE, digits_vel);
            if (digits_mass > 0)
                H5Pset_scaleoffset(h_prop_mass, H5Z_SO_FLOAT_DSCALE, digits_mass);
            if (digits_radius > 0)
                H5Pset_scaleoffset(h_prop_radius, H5Z_SO_FLOAT_DSCALE, digits_radius);

            /* Set shuffle and lossless compression filters (GZIP level 4) */
            const int gzip_level = pars->SnipshotZipCompressionLevel;
            if (gzip_level > 0) {
                H5Pset_shuffle(h_prop_pos);
                H5Pset_shuffle(h_prop_vel);
                H5Pset_shuffle(h_prop_mass);
                H5Pset_shuffle(h_prop_radius);
                H5Pset_deflate(h_prop_pos, gzip_level);
                H5Pset_deflate(h_prop_vel, gzip_level);
                H5Pset_deflate(h_prop_mass, gzip_level);
                H5Pset_deflate(h_prop_radius, gzip_level);
            }

            /* Create the particle group in the output file */
            h_grp = H5Gcreate(h_out_file, ExportNames[t], H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            
            /* Halo IDs (use scalar space) */
            h_data = H5Dcreate(h_grp, "ID", H5T_NATIVE_LLONG, h_sspace, H5P_DEFAULT, h_prop_sca, H5P_DEFAULT);
            H5Dclose(h_data);
            
            /* Centres of mass (use vector space) */
            h_data = H5Dcreate(h_grp, "CentreOfMass", H5T_NATIVE_FLOAT, h_vspace, H5P_DEFAULT, h_prop_pos, H5P_DEFAULT);
            H5Dclose(h_data);

            /* Shrinking sphere centre (use vector space) */
            h_data = H5Dcreate(h_grp, "ShrinkingSphereCentre", H5T_NATIVE_FLOAT, h_vspace, H5P_DEFAULT, h_prop_pos, H5P_DEFAULT);
            H5Dclose(h_data);

            /* Total mass (use scalar space) */
            h_data = H5Dcreate(h_grp, "Mass", H5T_NATIVE_FLOAT, h_sspace, H5P_DEFAULT, h_prop_mass, H5P_DEFAULT);
            H5Dclose(h_data);
            
            /* Numbers of particles (use scalar space) */
            h_data = H5Dcreate(h_grp, "ParticleNumber", H5T_NATIVE_INT, h_sspace, H5P_DEFAULT, h_prop_sca, H5P_DEFAULT);
            H5Dclose(h_data);

            /* Halo radius (use scalar space) */
            h_data = H5Dcreate(h_grp, "Radius", H5T_NATIVE_FLOAT, h_sspace, H5P_DEFAULT, h_prop_radius, H5P_DEFAULT);
            H5Dclose(h_data);

            /* The following properties are only for SO halos */
            if (t == 1) {
                /* Radius enclosing the innermost particles that determine the CoM for SO halos (use scalar space) */
                h_data = H5Dcreate(h_grp, "InnerRadius", H5T_NATIVE_FLOAT, h_sspace, H5P_DEFAULT, h_prop_radius, H5P_DEFAULT);
                H5Dclose(h_data);

                /* Centre of Mass velocities (use vector space) */
                h_data = H5Dcreate(h_grp, "CentreOfMassVelocity", H5T_NATIVE_FLOAT, h_vspace, H5P_DEFAULT, h_prop_vel, H5P_DEFAULT);
                H5Dclose(h_data);

                /* Shrinking sphere centre velocity (use vector space) */
                h_data = H5Dcreate(h_grp, "ShrinkingSphereVelocity", H5T_NATIVE_FLOAT, h_vspace, H5P_DEFAULT, h_prop_vel, H5P_DEFAULT);
                H5Dclose(h_data);

                /* Dark matter centres of mass (use vector space) */
                h_data = H5Dcreate(h_grp, "DarkMatterCentreOfMass", H5T_NATIVE_FLOAT, h_vspace, H5P_DEFAULT, h_prop_pos, H5P_DEFAULT);
                H5Dclose(h_data);

                /* Dark matter centre of mass velocities (use vector space) */
                h_data = H5Dcreate(h_grp, "DarkMatterCentreOfMassVelocity", H5T_NATIVE_FLOAT, h_vspace, H5P_DEFAULT, h_prop_vel, H5P_DEFAULT);
                H5Dclose(h_data);

                /* Total dark matter mass (use scalar space) */
                h_data = H5Dcreate(h_grp, "DarkMatterMass", H5T_NATIVE_FLOAT, h_sspace, H5P_DEFAULT, h_prop_mass, H5P_DEFAULT);
                H5Dclose(h_data);

                /* Total neutrino mass (use scalar space) */
                h_data = H5Dcreate(h_grp, "NeutrinoMass", H5T_NATIVE_FLOAT, h_sspace, H5P_DEFAULT, h_prop_mass, H5P_DEFAULT);
                H5Dclose(h_data);

                /* Total particle mass (use scalar space) */
                h_data = H5Dcreate(h_grp, "TotalParticleMass", H5T_NATIVE_FLOAT, h_sspace, H5P_DEFAULT, h_prop_mass, H5P_DEFAULT);
                H5Dclose(h_data);
            }
            
            /* Close the group */
            H5Gclose(h_grp);
        }

        /* Close the file */
        H5Fclose(h_out_file);        
    }

    /* Wait until all ranks are finished */
    MPI_Barrier(MPI_COMM_WORLD);
    

    /* Property list for MPI file access */
    hid_t prop_faxs = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(prop_faxs, MPI_COMM_WORLD, MPI_INFO_NULL);

    /* Open the hdf5 file */
    hid_t h_out_file = H5Fopen(fname, H5F_ACC_RDWR, prop_faxs);
    H5Pclose(prop_faxs);

    /* Create vector & scalar dataspace for all data of this type */
    const hsize_t vrank = 2;
    const hsize_t srank = 1;
    const hsize_t vdims[2] = {total_num_structures, 3};
    const hsize_t sdims[1] = {total_num_structures};
    hid_t h_vspace = H5Screate_simple(vrank, vdims, NULL);
    hid_t h_sspace = H5Screate_simple(srank, sdims, NULL);

    /* Set chunking for vectors */
    hid_t h_prop_vec = H5Pcreate(H5P_DATASET_CREATE);
    const hsize_t vchunk[2] = {HDF5_CHUNK_SIZE, 3};
    H5Pset_chunk(h_prop_vec, vrank, vchunk);

    /* Set chunking for scalars */
    hid_t h_prop_sca = H5Pcreate(H5P_DATASET_CREATE);
    const hsize_t schunk[1] = {HDF5_CHUNK_SIZE};
    H5Pset_chunk(h_prop_sca, srank, schunk);

    /* Create vector & scalar datapsace for smaller chunks of data */
    const hsize_t ch_vdims[2] = {local_num_structures, 3};
    const hsize_t ch_sdims[1] = {local_num_structures};
    hid_t h_ch_vspace = H5Screate_simple(vrank, ch_vdims, NULL);
    hid_t h_ch_sspace = H5Screate_simple(srank, ch_sdims, NULL);

    /* Determine the number of structures on each rank */
    long long int *halonum_by_rank = calloc(MPI_Rank_Count, sizeof(long long int));
    long long int *first_id_by_rank = calloc(MPI_Rank_Count, sizeof(long long int));
    halonum_by_rank[rank] = local_num_structures;
    MPI_Allreduce(MPI_IN_PLACE, halonum_by_rank, MPI_Rank_Count, MPI_LONG_LONG,
                  MPI_SUM, MPI_COMM_WORLD);

    /* Determine the start of the hyperslab corresponding to each rank */
    for (int i = 1; i < MPI_Rank_Count; i++) {
        first_id_by_rank[i] = first_id_by_rank[i - 1] + halonum_by_rank[i - 1];
    }

    /* The start of this chunk, in the overall vector & scalar spaces */
    const hsize_t start_in_group = first_id_by_rank[rank];
    const hsize_t vstart[2] = {start_in_group, 0}; //always with the "x" coordinate
    const hsize_t sstart[1] = {start_in_group};

    /* Free memory */
    free(halonum_by_rank);
    free(first_id_by_rank);

    /* Choose the corresponding hyperslabs inside the overall spaces */
    H5Sselect_hyperslab(h_vspace, H5S_SELECT_SET, vstart, NULL, ch_vdims, NULL);
    H5Sselect_hyperslab(h_sspace, H5S_SELECT_SET, sstart, NULL, ch_sdims, NULL);

    /* Unpack the halo data into contiguous arrays */
    long long int *ids = malloc(3 * local_num_structures * sizeof(long long int));
    float *coms = malloc(3 * local_num_structures * sizeof(float));
    float *coms_inner = malloc(3 * local_num_structures * sizeof(float));
    float *masses = malloc(1 * local_num_structures * sizeof(float));
    float *radii = malloc(1 * local_num_structures * sizeof(float));
    int *nparts = malloc(3 * local_num_structures * sizeof(int));
    for (long long i = 0; i < local_num_structures; i++) {
        struct fof_halo *h = &fofs[i];
        /* Unpack the particle IDs */
        ids[i] = h->global_id;
        /* Unpack the CoM coordinates */
        coms[i * 3 + 0] = h->x_com[0];
        coms[i * 3 + 1] = h->x_com[1];
        coms[i * 3 + 2] = h->x_com[2];
        /* Unpack the shrinking sphere coordinates */
        coms_inner[i * 3 + 0] = h->x_com_inner[0];
        coms_inner[i * 3 + 1] = h->x_com_inner[1];
        coms_inner[i * 3 + 2] = h->x_com_inner[2];
        /* Unpack the masses and radii */
        masses[i] = h->mass_fof;
        radii[i] = h->radius_fof;
        /* Unpack the particle numbers */
        nparts[i] = h->npart;
    }

    /* Open the particle group in the output file */
    hid_t h_grp = H5Gopen(h_out_file, ExportNames[0], H5P_DEFAULT);

    /* Property list for collective MPI write */
    hid_t prop_write = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(prop_write, H5FD_MPIO_COLLECTIVE);

    /* Write halo id data (scalar) */
    hid_t h_data = H5Dopen(h_grp, "ID", H5P_DEFAULT);
    H5Dwrite(h_data, H5T_NATIVE_LLONG, h_ch_sspace, h_sspace, prop_write, ids);
    H5Dclose(h_data);
    free(ids);

    /* Write centre of mass data (vector) */
    h_data = H5Dopen(h_grp, "CentreOfMass", H5P_DEFAULT);
    H5Dwrite(h_data, H5T_NATIVE_FLOAT, h_ch_vspace, h_vspace, prop_write, coms);
    H5Dclose(h_data);
    free(coms);

    /* Write mass data (scalar) */
    h_data = H5Dopen(h_grp, "Mass", H5P_DEFAULT);
    H5Dwrite(h_data, H5T_NATIVE_FLOAT, h_ch_sspace, h_sspace, prop_write, masses);
    H5Dclose(h_data);
    free(masses);

    /* Write particle number data (scalar) */
    h_data = H5Dopen(h_grp, "ParticleNumber", H5P_DEFAULT);
    H5Dwrite(h_data, H5T_NATIVE_INT, h_ch_sspace, h_sspace, prop_write, nparts);
    H5Dclose(h_data);
    free(nparts);

    /* Write radius data (scalar) */
    h_data = H5Dopen(h_grp, "Radius", H5P_DEFAULT);
    H5Dwrite(h_data, H5T_NATIVE_FLOAT, h_ch_sspace, h_sspace, prop_write, radii);
    H5Dclose(h_data);
    free(radii);

    /* Write centre of mass data (vector) */
    h_data = H5Dopen(h_grp, "ShrinkingSphereCentre", H5P_DEFAULT);
    H5Dwrite(h_data, H5T_NATIVE_FLOAT, h_ch_vspace, h_vspace, prop_write, coms_inner);
    H5Dclose(h_data);
    free(coms_inner);

    /* Close the property list and group */
    H5Pclose(prop_write);
    H5Gclose(h_grp);

    /* Close the file */
    H5Fclose(h_out_file);

    return 0;
}


int exportSOCatalogue(const struct params *pars, const struct units *us,
                      const struct physical_consts *pcs, int output_num, double a,
                      long int total_num_structures, long int local_num_structures,
                      struct so_halo *so_halos) {

    /* Get the dimensions of the cluster */
    int rank, MPI_Rank_Count;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &MPI_Rank_Count);

    /* Form the filename */
    char fname[DEFAULT_STRING_LENGTH];
    sprintf(fname, "%s_%04d.hdf5", pars->CatalogueBaseName, output_num);

    /* The HDF5 group name of each halo type */
    const char ExportName[100] = "SO_200_crit";

    /* Property list for MPI file access */
    hid_t prop_faxs = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(prop_faxs, MPI_COMM_WORLD, MPI_INFO_NULL);

    /* Open the hdf5 file */
    hid_t h_out_file = H5Fopen(fname, H5F_ACC_RDWR, prop_faxs);
    H5Pclose(prop_faxs);

    /* Create vector & scalar dataspace for all data of this type */
    const hsize_t vrank = 2;
    const hsize_t srank = 1;
    const hsize_t vdims[2] = {total_num_structures, 3};
    const hsize_t sdims[1] = {total_num_structures};
    hid_t h_vspace = H5Screate_simple(vrank, vdims, NULL);
    hid_t h_sspace = H5Screate_simple(srank, sdims, NULL);

    /* Set chunking for vectors */
    hid_t h_prop_vec = H5Pcreate(H5P_DATASET_CREATE);
    const hsize_t vchunk[2] = {HDF5_CHUNK_SIZE, 3};
    H5Pset_chunk(h_prop_vec, vrank, vchunk);

    /* Set chunking for scalars */
    hid_t h_prop_sca = H5Pcreate(H5P_DATASET_CREATE);
    const hsize_t schunk[1] = {HDF5_CHUNK_SIZE};
    H5Pset_chunk(h_prop_sca, srank, schunk);

    /* Create vector & scalar datapsace for smaller chunks of data */
    const hsize_t ch_vdims[2] = {local_num_structures, 3};
    const hsize_t ch_sdims[1] = {local_num_structures};
    hid_t h_ch_vspace = H5Screate_simple(vrank, ch_vdims, NULL);
    hid_t h_ch_sspace = H5Screate_simple(srank, ch_sdims, NULL);

    /* Determine the number of structures on each rank */
    long long int *halonum_by_rank = calloc(MPI_Rank_Count, sizeof(long long int));
    long long int *first_id_by_rank = calloc(MPI_Rank_Count, sizeof(long long int));
    halonum_by_rank[rank] = local_num_structures;
    MPI_Allreduce(MPI_IN_PLACE, halonum_by_rank, MPI_Rank_Count, MPI_LONG_LONG,
                  MPI_SUM, MPI_COMM_WORLD);

    /* Determine the start of the hyperslab corresponding to each rank */
    for (int i = 1; i < MPI_Rank_Count; i++) {
        first_id_by_rank[i] = first_id_by_rank[i - 1] + halonum_by_rank[i - 1];
    }

    /* The start of this chunk, in the overall vector & scalar spaces */
    const hsize_t start_in_group = first_id_by_rank[rank];
    const hsize_t vstart[2] = {start_in_group, 0}; //always with the "x" coordinate
    const hsize_t sstart[1] = {start_in_group};

    /* Free memory */
    free(halonum_by_rank);
    free(first_id_by_rank);

    /* Choose the corresponding hyperslabs inside the overall spaces */
    H5Sselect_hyperslab(h_vspace, H5S_SELECT_SET, vstart, NULL, ch_vdims, NULL);
    H5Sselect_hyperslab(h_sspace, H5S_SELECT_SET, sstart, NULL, ch_sdims, NULL);

    /* Unpack the halo data into contiguous arrays */
    long long int *ids = malloc(3 * local_num_structures * sizeof(long long int));
    float *coms = malloc(3 * local_num_structures * sizeof(float));
    float *coms_inner = malloc(3 * local_num_structures * sizeof(float));
    float *coms_dm = malloc(3 * local_num_structures * sizeof(float));
    float *vels = malloc(3 * local_num_structures * sizeof(float));
    float *vels_inner = malloc(3 * local_num_structures * sizeof(float));
    float *vels_dm = malloc(3 * local_num_structures * sizeof(float));
    float *masses = malloc(1 * local_num_structures * sizeof(float));
    float *masses_dm = malloc(1 * local_num_structures * sizeof(float));
    float *masses_nu = malloc(1 * local_num_structures * sizeof(float));
    float *masses_tot = malloc(1 * local_num_structures * sizeof(float));
    float *radii = malloc(1 * local_num_structures * sizeof(float));
    float *inner_radii = malloc(1 * local_num_structures * sizeof(float));
    int *nparts = malloc(3 * local_num_structures * sizeof(int));
    for (long long i = 0; i < local_num_structures; i++) {
        struct so_halo *h = &so_halos[i];
        /* Unpack the particle IDs */
        ids[i] = h->global_id;
        /* Unpack the CoM coordinates */
        coms[i * 3 + 0] = h->x_com[0];
        coms[i * 3 + 1] = h->x_com[1];
        coms[i * 3 + 2] = h->x_com[2];
        /* Unpack the shrinking sphere coordinates */
        coms_inner[i * 3 + 0] = h->x_com_inner[0];
        coms_inner[i * 3 + 1] = h->x_com_inner[1];
        coms_inner[i * 3 + 2] = h->x_com_inner[2];
        /* Unpack the dark matter CoM coordinates */
        coms_dm[i * 3 + 0] = h->x_com_dm[0];
        coms_dm[i * 3 + 1] = h->x_com_dm[1];
        coms_dm[i * 3 + 2] = h->x_com_dm[2];
        /* Unpack the CoM velocities */
        vels[i * 3 + 0] = h->v_com[0];
        vels[i * 3 + 1] = h->v_com[1];
        vels[i * 3 + 2] = h->v_com[2];
        /* Unpack the shrinking sphere velocities */
        vels_inner[i * 3 + 0] = h->v_com_inner[0];
        vels_inner[i * 3 + 1] = h->v_com_inner[1];
        vels_inner[i * 3 + 2] = h->v_com_inner[2];
        /* Unpack the dark matter CoM velocities */
        vels_dm[i * 3 + 0] = h->v_com_dm[0];
        vels_dm[i * 3 + 1] = h->v_com_dm[1];
        vels_dm[i * 3 + 2] = h->v_com_dm[2];
        /* Unpack the SO masses and radii */
        masses[i] = h->M_SO;
        masses_dm[i] = h->mass_dm;
        masses_nu[i] = h->mass_nu;
        masses_tot[i] = h->mass_tot;
        radii[i] = h->R_SO;
        inner_radii[i] = h->R_inner;
        /* Unpack the particle numbers */
        nparts[i] = h->npart_tot;
    }

    /* Open the particle group in the output file */
    hid_t h_grp = H5Gopen(h_out_file, ExportName, H5P_DEFAULT);

    /* Property list for collective MPI write */
    hid_t prop_write = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(prop_write, H5FD_MPIO_COLLECTIVE);

    /* Write halo id data (scalar) */
    hid_t h_data = H5Dopen(h_grp, "ID", H5P_DEFAULT);
    H5Dwrite(h_data, H5T_NATIVE_LLONG, h_ch_sspace, h_sspace, prop_write, ids);
    H5Dclose(h_data);
    free(ids);

    /* Write centre of mass data (vector) */
    h_data = H5Dopen(h_grp, "CentreOfMass", H5P_DEFAULT);
    H5Dwrite(h_data, H5T_NATIVE_FLOAT, h_ch_vspace, h_vspace, prop_write, coms);
    H5Dclose(h_data);
    free(coms);

    /* Write innermost centre of mass data (vector) */
    h_data = H5Dopen(h_grp, "ShrinkingSphereCentre", H5P_DEFAULT);
    H5Dwrite(h_data, H5T_NATIVE_FLOAT, h_ch_vspace, h_vspace, prop_write, coms_inner);
    H5Dclose(h_data);
    free(coms_inner);

    /* Write dark matter centre of mass data (vector) */
    h_data = H5Dopen(h_grp, "DarkMatterCentreOfMass", H5P_DEFAULT);
    H5Dwrite(h_data, H5T_NATIVE_FLOAT, h_ch_vspace, h_vspace, prop_write, coms_dm);
    H5Dclose(h_data);
    free(coms_dm);

    /* Write centre of mass velocity data (vector) */
    h_data = H5Dopen(h_grp, "CentreOfMassVelocity", H5P_DEFAULT);
    H5Dwrite(h_data, H5T_NATIVE_FLOAT, h_ch_vspace, h_vspace, prop_write, vels);
    H5Dclose(h_data);
    free(vels);

    /* Write innermost centre of mass velocity data (vector) */
    h_data = H5Dopen(h_grp, "ShrinkingSphereVelocity", H5P_DEFAULT);
    H5Dwrite(h_data, H5T_NATIVE_FLOAT, h_ch_vspace, h_vspace, prop_write, vels_inner);
    H5Dclose(h_data);
    free(vels_inner);

    /* Write dark matter centre of mass velocity data (vector) */
    h_data = H5Dopen(h_grp, "DarkMatterCentreOfMassVelocity", H5P_DEFAULT);
    H5Dwrite(h_data, H5T_NATIVE_FLOAT, h_ch_vspace, h_vspace, prop_write, vels_dm);
    H5Dclose(h_data);
    free(vels_dm);

    /* Write mass data (scalar) */
    h_data = H5Dopen(h_grp, "Mass", H5P_DEFAULT);
    H5Dwrite(h_data, H5T_NATIVE_FLOAT, h_ch_sspace, h_sspace, prop_write, masses);
    H5Dclose(h_data);
    free(masses);

    /* Write dark matter mass data (scalar) */
    h_data = H5Dopen(h_grp, "DarkMatterMass", H5P_DEFAULT);
    H5Dwrite(h_data, H5T_NATIVE_FLOAT, h_ch_sspace, h_sspace, prop_write, masses_dm);
    H5Dclose(h_data);
    free(masses_dm);

    /* Write neutrino mass data (scalar) */
    h_data = H5Dopen(h_grp, "NeutrinoMass", H5P_DEFAULT);
    H5Dwrite(h_data, H5T_NATIVE_FLOAT, h_ch_sspace, h_sspace, prop_write, masses_nu);
    H5Dclose(h_data);
    free(masses_nu);

    /* Write total particle mass data (scalar) */
    h_data = H5Dopen(h_grp, "TotalParticleMass", H5P_DEFAULT);
    H5Dwrite(h_data, H5T_NATIVE_FLOAT, h_ch_sspace, h_sspace, prop_write, masses_tot);
    H5Dclose(h_data);
    free(masses_tot);

    /* Write radius data (scalar) */
    h_data = H5Dopen(h_grp, "Radius", H5P_DEFAULT);
    H5Dwrite(h_data, H5T_NATIVE_FLOAT, h_ch_sspace, h_sspace, prop_write, radii);
    H5Dclose(h_data);
    free(radii);

    /* Write inner radius data (scalar) */
    h_data = H5Dopen(h_grp, "InnerRadius", H5P_DEFAULT);
    H5Dwrite(h_data, H5T_NATIVE_FLOAT, h_ch_sspace, h_sspace, prop_write, inner_radii);
    H5Dclose(h_data);
    free(inner_radii);

    /* Write particle number data (scalar) */
    h_data = H5Dopen(h_grp, "ParticleNumber", H5P_DEFAULT);
    H5Dwrite(h_data, H5T_NATIVE_INT, h_ch_sspace, h_sspace, prop_write, nparts);
    H5Dclose(h_data);
    free(nparts);

    /* Close the property list and group */
    H5Pclose(prop_write);
    H5Gclose(h_grp);

    /* Close the file */
    H5Fclose(h_out_file);

    return 0;
}