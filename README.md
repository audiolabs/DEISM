# Diffraction Enhanced Image Source Method

The code in this folder is able to solve the following problem: 

A source and a receiver transducer with arbitrary directivity are mounted on one/two speakers; The local scattering and diffraction effects around the transducers result in complex directivity patterns. The directivity patterns can be obtained by analytical expressions, numerical simulations or measurements. 

In DEISM, we can model the room transfer function between transducers mounted on one/two speakers using the image source method while incorporating the local diffraction effects around the transducers. The local diffraction effects are captured using spherical-harmonic directivity coefficients obtained on a sphere around the transducers.

## Usage

- Use command "conda env create -f DEISM.yml" to create an conda environment for the DEISM algorithms.
- Run script "DEISM_example.py".

## Tips

The example code only provide essential functionalities based on DEISM. For more complex scenarios, please contact the authors (zeyu.xu@audiolabs-erlangen.de) for support. In the following, a few important tips might be helpful for you: 

- If you want to simulate the scenario where both the source and receiver are positioned on the same speaker, you need to run DEISM for all reflection path except for the direct path. To do this, just remove the path with index q_x,q_y,q_z,p_x,p_y,p_z=0,0,0,0,0,0 from variable A and all the other related variables
- It is recommended to set the distance at least 1m between the transducers and the walls. 
- If you want to use your own directivity data, please check the function [load_segdata_src_rec](# Function load_segdata_src_rec) and [load_segdata_room](# Function load_segdata_room) in detail. 



# Description of the codes and functions

## DEISM_example.py 

A basic example of the simulation setups. This script let you set the parameters as you need and then runs DEISM and DEISM LC as a comparison. In the end, their magnitudes and phases are plotted along with those of the FEM simulation. 

### Parameters

Note that the parameters are stored in a dictionary called "params" in the codes. If a parameter is marked by a ''*'',  it means that it requires some further thoughts while dealing with specific problems. 

| NAME                | description and notes                                        |
| ------------------- | ------------------------------------------------------------ |
| c                   | Speed of sound                                               |
| LL                  | Room dimension                                               |
| N_o                 | Max. reflection order                                        |
| fs*                 | Sampling rate. Need to specify if reverberation time is known. |
| RT60*               | Reverberation time. The specific formula has not been implemented, use max. reflection order instead. |
| fstart              | Start frequency                                              |
| fstep               | Frequency step size                                          |
| fstop               | Maximum frequency                                            |
| k                   | Wavenumbers                                                  |
| Z_S                 | Acoustic impedance of the walls                              |
| beta_Refcoef        | Reflection coefficients of the walls if they are independent of the incident angles |
| RefCoef_angdep_flag | A flag determining if angle-dependent reflection coefficients are used |
| Q*                  | Point source flow strength used in FEM simulation. Please change it if you use your own simulation data, e.g., directivity or room transfer functions. |
| rho0*               | Density of air                                               |
| S*                  | Point source strength. One probably need this to normalize the directivity or transfer function if obtained using FEM simulation. |
| src_rec_type*       | Define the speaker type. It is only related to the file names of the directivity data. |
| num_samples*        | Number of sampling points of the directivity patterns. Only used to load the data. |
| sampling_scheme*    | Sampling scheme of the directivity patterns. Only used to load the data. |
| x_s                 | Location of source transducer in Cartesian coordinates.      |
| src_facing          | the facing of the source speaker, usually it is the direction that the circular piston is facing. |
| N_src_dir           | Maximum spherical harmonic order used to represent the source directivity. |
| r0_src              | Radius of the sphere around the source transducer where the directivity is sampled |
| x_r                 | Location of receiver transducer in Cartesian coordinates.    |
| rec_facing          | the facing of the receiver speaker, usually it is the direction that the circular piston is facing. |
| V_rec_dir           | Maximum spherical harmonic order used to represent the receiver directivity. |
| r0_rec              | Radius of the sphere around the receiver transducer where the directivity is sampled |

## helper.py

### Function pre_calc_Wigner

Precalculation of Wigner-3j symbols as matrices to avoid long computation in each loop of the images. 

### Function pre_calc_images_src_rec

Precalculation of the images of the source and receiver, reflection paths, attenuation of the each reflection paths. 

### Functions T_x, T_y, T_z

Transformation matrices to determine the effect of successive reflections on different walls in a reflection path. See paper "Fast Source-Room-Receiver Acoustics Modeling" by Y. Luo and W. Kim. 

### Function rotation_matrix_ZXZ

Rotation matrices given Euler angles using "x-convention". [^Euler]

### Function SHCs_from_pressure_LS

Obtain spherical harmonic coefficients from the simulated (or measured) sound field on a sphere with a fixed radius. The coefficients are calculated using the Least-Square solution of the sampling weights. 

### Function get_C_nm_s

Get spherical harmonic directivity coefficients of the source transducer from the spherical harmonic coefficients obtained by function SHCs_from_pressure_LS. 

### Function get_C_vu_r

Get spherical harmonic directivity coefficients of the receiver transducer from the spherical harmonic coefficients obtained by function SHCs_from_pressure_LS. 

### Function calc_DEISM_FEM_single_reflection

DEISM calculation of one reflection path. 

### Function ray_run_DEISM

Parallel DEISM calculation of all reflection paths. 

### Function calc_DEISM_FEM_simp_single_reflection

DEISM LC calculation of one reflection path.

### Function ray_run_DEISM_LC

Parallel calculation of DEISM LC for all reflection paths. 

### Function load_segdata_src_rec

Load simulated (or measured) sound pressure data around a transducer. 

### Function load_segdata_room

Load simulated (or measured) room transfer function from a source to a receiver. 



[^Euler]: https://mathworld.wolfram.com/EulerAngles.html
