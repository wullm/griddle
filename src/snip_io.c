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

/* Methods for input and output of particle snapshots */

#include <stdlib.h>
#include <string.h>
#include <hdf5.h>
#include <assert.h>
#include <math.h>
#include "../include/snip_io.h"
#include "../include/analysis_so.h"
#include "../include/relativity.h"
#include "../include/git_version.h"

/* The current limit of for parallel HDF5 writes is 2GB */
#define HDF5_PARALLEL_LIMIT 2147000000LL
/* The default chunk size, corresponding to ~0.5 MB */
#define HDF5_CHUNK_SIZE 65536LL
/* TODO: Not the default chunk size, decide what makes sense */
#define HDF5_TINY_CHUNK_SIZE 128LL

int writeSnipshotHeader(const struct params *pars, const struct units *us,
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

    /* Create the Flag_Entropy_ICs attribute and write the data */
    int flag_entropy = 0;
    h_attr = H5Acreate1(h_grp, "Flag_Entropy_ICs", H5T_NATIVE_INT, h_aspace, H5P_DEFAULT);
    H5Awrite(h_attr, H5T_NATIVE_INT, &flag_entropy);
    H5Aclose(h_attr);

    /* Create the NumFilesPerSnapshot attribute and write the data */
    int num_files_per_snapshot = 1;
    h_attr = H5Acreate1(h_grp, "NumFilesPerSnapshot", H5T_NATIVE_INT, h_aspace, H5P_DEFAULT);
    H5Awrite(h_attr, H5T_NATIVE_INT, &num_files_per_snapshot);
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

    /* Change dataspace dimensions to particle type attributes */
    const hsize_t adims_pt[1] = {7}; //particle type 0-6
    H5Sset_extent_simple(h_aspace, arank, adims_pt, NULL);

    /* Unused attribute for backwards compatibility */
    double mass_table[7];
    for (int i = 0; i < 7; i++) {
        mass_table[i] = 0.;
    }

    /* Create the MassTable attribute and write the data */
    h_attr = H5Acreate1(h_grp, "MassTable", H5T_NATIVE_DOUBLE, h_aspace, H5P_DEFAULT);
    H5Awrite(h_attr, H5T_NATIVE_DOUBLE, mass_table);
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

int exportSnipshot(const struct params *pars, const struct units *us,
                   const struct so_halo *halos, const struct physical_consts *pcs,
                   const struct particle *parts, const struct cosmology *cosmo,
                   const struct so_cell_list *cell_list, long int *cell_counts,
                   long int *cell_offsets, int output_num, double a_scale_factor,
                   CellIntType N_cells, double reduce_factor,
                   int min_part_export_per_halo, long long int local_partnum,
                   long long int local_halo_num, double dtau_kick,
                   double dtau_drift) {

    /* Get the dimensions of the cluster */
    int rank, MPI_Rank_Count;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &MPI_Rank_Count);
    
    /* Communicate the halo counts across all ranks */
    long long int *halos_per_rank = malloc(MPI_Rank_Count * sizeof(long int));
    long long int *halo_rank_offsets = malloc(MPI_Rank_Count * sizeof(long int));

    MPI_Allgather(&local_halo_num, 1, MPI_LONG_LONG, halos_per_rank, 1,
                  MPI_LONG_LONG, MPI_COMM_WORLD);

    halo_rank_offsets[0] = 0;
    for (int i = 1; i < MPI_Rank_Count; i++) {
        halo_rank_offsets[i] = halo_rank_offsets[i - 1] + halos_per_rank[i - 1];
    }

    const long long int total_halo_num = halo_rank_offsets[MPI_Rank_Count - 1] + halos_per_rank[MPI_Rank_Count - 1];

    /* Position factors */
    const double boxlen = pars->BoxLength;
    const double pos_to_int_fac = pow(2.0, POSITION_BITS) / boxlen;
    const double int_to_pos_fac = 1.0 / pos_to_int_fac;
    const double pos_to_cell_fac = N_cells / boxlen;
    
    /* First, conservatively estimate the number of particles to be exported */
    long int maximum_local_number = 0;
    
    /* Loop over halos */
    for (long int i = 0; i < local_halo_num; i++) {

        /* Determine the selection probability */
        double p_select;
        if (halos[i].npart_tot == 0) {
            continue; // no particles to expect here
        } else if (reduce_factor * halos[i].npart_tot < min_part_export_per_halo * 2) {
            p_select = 2.0 * min_part_export_per_halo / halos[i].npart_tot;
        } else {
            p_select = reduce_factor;
        }

        maximum_local_number += p_select * halos[i].npart_tot * 1.1;                               
    }

    /* Unpack the remaining particle data into contiguous arrays */
    double *coords = malloc(3 * maximum_local_number * sizeof(double));
    double *vels = malloc(3 * maximum_local_number * sizeof(double));
#ifdef WITH_PARTICLE_IDS
    long long *ids = malloc(1 * maximum_local_number * sizeof(long long));
#endif
#ifdef WITH_MASSES
    double *masses = malloc(1 * maximum_local_number * sizeof(double));
#endif

    /* Count the number of exported particles per halo and in total */
    int *particles_per_halo = calloc((int) local_halo_num, sizeof(int));
    long long int particles_total = 0;

    /* Memory for holding the indices of overlapping cells */
    CellIntType *cells = malloc(0);
    CellIntType num_overlap;

    /* Loop over SO halos */
    for (long int i = 0; i < local_halo_num; i++) {

        /* Compute the integer position of the halo CoM */
        IntPosType com[3] = {halos[i].x_com[0] * pos_to_int_fac,
                             halos[i].x_com[1] * pos_to_int_fac,
                             halos[i].x_com[2] * pos_to_int_fac};

        /* The square of the SO radius */
        double R_SO_2 = halos[i].R_SO * halos[i].R_SO;

        /* Determine the selection probability */
        double p_select;
        if (halos[i].npart_tot == 0) {
            continue; // no particles to expect here
        } else if (reduce_factor * halos[i].npart_tot < min_part_export_per_halo * 2) {
            p_select = 2.0 * min_part_export_per_halo / halos[i].npart_tot;
        } else {
            p_select = reduce_factor;
        }
        
        /* Determine all cells that overlap with the search radius */
        find_overlapping_cells(halos[i].x_com, halos[i].R_SO,
                               pos_to_cell_fac, N_cells, &cells, &num_overlap);

        /* Loop over cells */
        for (CellIntType c = 0; c < num_overlap; c++) {
            /* Find the particle count and offset of the cell */
            CellIntType cell = cells[c];
            long int local_count = cell_counts[cell];
            long int local_offset = cell_offsets[cell];
            
            /* Loop over particles in cells */
            for (long int a = 0; a < local_count; a++) {
                const long int index_a = cell_list[local_offset + a].offset;

                /* Skip non-DM particles */
                if (!match_particle_type(&parts[index_a], cdm_type, 1)) continue;

                const IntPosType *xa = parts[index_a].x;
                const double r2 = int_to_phys_dist2(xa, com, int_to_pos_fac);

                if (r2 < R_SO_2) {
                    /* Randomly decide whether to select */
                    double p = rand() / ((double) RAND_MAX);

                    if (p < p_select) {
                        /* Unpack the coordinates */
                        coords[particles_total * 3 + 0] = parts[index_a].x[0] * int_to_pos_fac;
                        coords[particles_total * 3 + 1] = parts[index_a].x[1] * int_to_pos_fac;
                        coords[particles_total * 3 + 2] = parts[index_a].x[2] * int_to_pos_fac;
                        /* Unpack the velocities */
                        vels[particles_total * 3 + 0] = parts[index_a].v[0];
                        vels[particles_total * 3 + 1] = parts[index_a].v[1];
                        vels[particles_total * 3 + 2] = parts[index_a].v[2];
#ifdef WITH_ACCELERATIONS
                        /* Kick to the right time */
                        if (dtau_kick != 0.) {
                            vels[particles_total * 3 + 0] += parts[index_a].a[0] * dtau_kick;
                            vels[particles_total * 3 + 1] += parts[index_a].a[1] * dtau_kick;
                            vels[particles_total * 3 + 2] += parts[index_a].a[2] * dtau_kick;
                        }
#endif
                        /* Convert internal velocities to peculiar velocities */
                        vels[particles_total * 3 + 0] /= a_scale_factor;
                        vels[particles_total * 3 + 1] /= a_scale_factor;
                        vels[particles_total * 3 + 2] /= a_scale_factor;
#ifdef WITH_PARTICLE_IDS
                        /* Unpack the particle IDs */
                        ids[particles_total] = parts[index_a].id;
#endif
#ifdef WITH_MASSES
                        /* Unpack the particle masses */
                        masses[particles_total] = parts[index_a].m;
#endif
                        particles_total++;
                        particles_per_halo[i]++;
                        
                        if (particles_total >= maximum_local_number) {
                            printf("Error: Not enough memory allocated for snipshot particles.\n");
                            exit(1);
                        }
                    }
                }
            }  /* End particle loop */
        } /* End cell loop */
    } /* End halo loop */
    
    /* Free the cell indices */
    free(cells);
    
    /* Reallocate the particle data arrays */
    coords = realloc(coords, 3 * particles_total * sizeof(double));
    vels = realloc(vels, 3 * particles_total * sizeof(double));
#ifdef WITH_PARTICLE_IDS
    ids = realloc(ids, 1 * particles_total * sizeof(long long));
#endif
#ifdef WITH_MASSES
    masses = realloc(masses, 1 * particles_total * sizeof(double));
#endif

    /* Determine the number of particles on each rank */
    long long int *partnum_by_rank = calloc(MPI_Rank_Count, sizeof(long long int));
    long long int *first_id_by_rank = calloc(MPI_Rank_Count, sizeof(long long int));
    long long int global_total_partnum = 0;
    partnum_by_rank[rank] = particles_total;
    MPI_Allreduce(MPI_IN_PLACE, partnum_by_rank, MPI_Rank_Count, MPI_LONG_LONG,
                  MPI_SUM, MPI_COMM_WORLD);

    /* Determine the start of the hyperslab corresponding to each rank */
    for (int i = 1; i < MPI_Rank_Count; i++) {
        first_id_by_rank[i] = first_id_by_rank[i - 1] + partnum_by_rank[i - 1];
    }
    /* Determine the total number of particles across all ranks */
    for (int i = 0; i < MPI_Rank_Count; i++) {
        global_total_partnum += partnum_by_rank[i];
    }
    
    /* Create vector & scalar dataspace for all data of this type */
    const hsize_t vrank = 2;
    const hsize_t srank = 1;
    const hsize_t vdims[2] = {global_total_partnum, 3};
    const hsize_t sdims[1] = {global_total_partnum};
    hid_t h_vspace = H5Screate_simple(vrank, vdims, NULL);
    hid_t h_sspace = H5Screate_simple(srank, sdims, NULL);

    /* Set chunking for vectors */
    hid_t h_prop_vec = H5Pcreate(H5P_DATASET_CREATE);
    const hsize_t vchunk[2] = {HDF5_TINY_CHUNK_SIZE, 3};
    H5Pset_chunk(h_prop_vec, vrank, vchunk);

    /* Set chunking for scalars */
    hid_t h_prop_sca = H5Pcreate(H5P_DATASET_CREATE);
    const hsize_t schunk[1] = {HDF5_TINY_CHUNK_SIZE};
    H5Pset_chunk(h_prop_sca, srank, schunk);

    /* Create vector & scalar datapsace for smaller chunks of data */
    const hsize_t ch_vdims[2] = {particles_total, 3};
    const hsize_t ch_sdims[1] = {particles_total};
    hid_t h_ch_vspace = H5Screate_simple(vrank, ch_vdims, NULL);
    hid_t h_ch_sspace = H5Screate_simple(srank, ch_sdims, NULL);

    /* The start of this chunk, in the overall vector & scalar spaces */
    const hsize_t start_in_group = first_id_by_rank[rank];
    const hsize_t vstart[2] = {start_in_group, 0}; //always with the "x" coordinate
    const hsize_t sstart[1] = {start_in_group};
    
    /* Free memory */
    free(partnum_by_rank);
    free(first_id_by_rank);
    
    /* The HDF5 group name of the dark matter particle type */
    const char ExportName[100] = "PartType1";

    /* Form the filename */
    char fname[DEFAULT_STRING_LENGTH];
    sprintf(fname, "%s_%04d.hdf5", pars->SnipBaseName, output_num);
    
    if (rank == 0) {
        /* Create the output file */
        hid_t h_out_file = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

        /* Writing attributes into the Header & Cosmology groups */
        int err = writeSnipshotHeader(pars, us, a_scale_factor, h_out_file);
        if (err > 0) exit(1);

        /* The particle group in the output file */
        hid_t h_grp;

        /* Datsets */
        hid_t h_data;
        
        /* Convert to SWIFT/GADGET style particle number tables */
        long long int numparts_local[7] = {0, global_total_partnum, 0, 0, 0, 0, 0};
        long long int numparts_total[7] = {0, global_total_partnum, 0, 0, 0, 0, 0};
        
        /* Open the header group in the output file */
        h_grp = H5Gopen(h_out_file, "Header", H5P_DEFAULT);
        
        /* Change dataspace dimensions to particle type attributes */
        const hsize_t arank = 1;
        const hsize_t adims_pt[1] = {7}; //particle type 0-6
        hid_t h_aspace = H5Screate_simple(arank, adims_pt, NULL);

        /* Attribute for backwards compatibility */
        long long int numparts_high_word[7];
        for (int i = 0; i < 7; i++) {
            numparts_high_word[i] = numparts_total[i] >> 32;
        }

        /* Create the NumPart_ThisFile attribute and write the data */
        hid_t h_attr = H5Acreate1(h_grp, "NumPart_ThisFile", H5T_NATIVE_LONG, h_aspace, H5P_DEFAULT);
        H5Awrite(h_attr, H5T_NATIVE_LONG, numparts_local);
        H5Aclose(h_attr);

        /* Create the NumPart_Total attribute and write the data */
        h_attr = H5Acreate1(h_grp, "NumPart_Total", H5T_NATIVE_LONG, h_aspace, H5P_DEFAULT);
        H5Awrite(h_attr, H5T_NATIVE_LONG, numparts_total);
        H5Aclose(h_attr);

        /* Create the NumPart_Total_HighWord attribute and write the data */
        h_attr = H5Acreate1(h_grp, "NumPart_Total_HighWord", H5T_NATIVE_LONG, h_aspace, H5P_DEFAULT);
        H5Awrite(h_attr, H5T_NATIVE_LONG, numparts_high_word);
        H5Aclose(h_attr);
        
        /* Close the header group */
        H5Gclose(h_grp);

        /* Create the particle group in the output file */
        h_grp = H5Gcreate(h_out_file, ExportName, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        /* Coordinates (use vector space) */
        h_data = H5Dcreate(h_grp, "Coordinates", H5T_NATIVE_DOUBLE, h_vspace, H5P_DEFAULT, h_prop_vec, H5P_DEFAULT);
        H5Dclose(h_data);

        /* Velocities (use vector space) */
        h_data = H5Dcreate(h_grp, "Velocities", H5T_NATIVE_DOUBLE, h_vspace, H5P_DEFAULT, h_prop_vec, H5P_DEFAULT);
        H5Dclose(h_data);

#ifdef WITH_PARTICLE_IDS
        /* Particle IDs (use scalar space) */
        h_data = H5Dcreate(h_grp, "ParticleIDs", H5T_NATIVE_LLONG, h_sspace, H5P_DEFAULT, h_prop_sca, H5P_DEFAULT);
        H5Dclose(h_data);
#endif
#ifdef WITH_MASSES
        /* Masses (use scalar space) */
        h_data = H5Dcreate(h_grp, "Masses", H5T_NATIVE_DOUBLE, h_sspace, H5P_DEFAULT, h_prop_sca, H5P_DEFAULT);
        H5Dclose(h_data);
#endif

        /* Close the group */
        H5Gclose(h_grp);
        
        /* Scalar dataspace for halos */
        const hsize_t hsdims[1] = {total_halo_num};
        hid_t h_hsspace = H5Screate_simple(srank, hsdims, NULL);
        
        /* Set chunking for halo scalars */
        hid_t h_prop_hsca = H5Pcreate(H5P_DATASET_CREATE);
        const hsize_t hschunk[1] = {HDF5_TINY_CHUNK_SIZE};
        H5Pset_chunk(h_prop_hsca, srank, hschunk);

        /* Create the halo group in the output file */
        h_grp = H5Gcreate(h_out_file, "Halos", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        /* Coordinates (use vector space) */
        h_data = H5Dcreate(h_grp, "ParticleCount", H5T_NATIVE_INT, h_hsspace, H5P_DEFAULT, h_prop_hsca, H5P_DEFAULT);
        H5Dclose(h_data);
        
        /* Close the halo group */
        H5Gclose(h_grp);

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
    
    /* Choose the hyperslabs for the local particles inside the overall spaces */
    H5Sselect_hyperslab(h_vspace, H5S_SELECT_SET, vstart, NULL, ch_vdims, NULL);
    H5Sselect_hyperslab(h_sspace, H5S_SELECT_SET, sstart, NULL, ch_sdims, NULL);

    /* Open the particle group in the output file */
    hid_t h_grp = H5Gopen(h_out_file, ExportName, H5P_DEFAULT);

    /* Write coordinate data (vector) */
    hid_t h_data = H5Dopen(h_grp, "Coordinates", H5P_DEFAULT);
    H5Dwrite(h_data, H5T_NATIVE_DOUBLE, h_ch_vspace, h_vspace, H5P_DEFAULT, coords);
    H5Dclose(h_data);
    free(coords);

    /* Write velocity data (vector) */
    h_data = H5Dopen(h_grp, "Velocities", H5P_DEFAULT);
    H5Dwrite(h_data, H5T_NATIVE_DOUBLE, h_ch_vspace, h_vspace, H5P_DEFAULT, vels);
    H5Dclose(h_data);
    free(vels);

#ifdef WITH_PARTICLE_IDS
    /* Write particle id data (scalar) */
    h_data = H5Dopen(h_grp, "ParticleIDs", H5P_DEFAULT);
    H5Dwrite(h_data, H5T_NATIVE_LLONG, h_ch_sspace, h_sspace, H5P_DEFAULT, ids);
    H5Dclose(h_data);
    free(ids);
#endif

#ifdef WITH_MASSES
    /* Write mass data (scalar) */
    h_data = H5Dopen(h_grp, "Masses", H5P_DEFAULT);
    H5Dwrite(h_data, H5T_NATIVE_DOUBLE, h_ch_sspace, h_sspace, H5P_DEFAULT, masses);
    H5Dclose(h_data);
    free(masses);
#endif

    /* Close the group */
    H5Gclose(h_grp);
    
    /* Open the halo group in the output file */
    h_grp = H5Gopen(h_out_file, "Halos", H5P_DEFAULT);
    
    /* Create scalar dataspace for halos */
    const hsize_t hsdims[1] = {total_halo_num};
    hid_t h_hsspace = H5Screate_simple(srank, hsdims, NULL);

    /* Set chunking for halo scalars */
    hid_t h_prop_hsca = H5Pcreate(H5P_DATASET_CREATE);
    const hsize_t hschunk[1] = {HDF5_TINY_CHUNK_SIZE};
    H5Pset_chunk(h_prop_hsca, srank, hschunk);

    /* Create halo scalar datapsace for smaller chunks of data */
    const hsize_t ch_hsdims[1] = {local_halo_num};
    hid_t h_ch_hsspace = H5Screate_simple(srank, ch_hsdims, NULL);

    /* The start of this chunk, in the overall halo scalar space */
    const hsize_t hsstart[1] = {halo_rank_offsets[rank]};
    
    /* Select the hyperslab corresponding to the local halos */
    H5Sselect_hyperslab(h_hsspace, H5S_SELECT_SET, hsstart, NULL, ch_hsdims, NULL);
    
    /* Write particle count data (scalar) */
    h_data = H5Dopen(h_grp, "ParticleCount", H5P_DEFAULT);
    H5Dwrite(h_data, H5T_NATIVE_INT, h_ch_hsspace, h_hsspace, H5P_DEFAULT, particles_per_halo);
    H5Dclose(h_data);
    free(particles_per_halo);
    
    /* Close the halo group */
    H5Gclose(h_grp);
    
    /* Close the file */
    H5Fclose(h_out_file);

    /* We are done with halo rank offsets and sizes */
    free(halos_per_rank);
    free(halo_rank_offsets);

    return 0;
}
