# Diffraction Enhanced Image Source Method - Arbitrary Room Geometry (DEISM-ARG)

[![DOI](https://zenodo.org/badge/666336301.svg)](https://doi.org/10.5281/zenodo.14055865)

The code in this folder is able to solve the following problem: 

A source and a receiver transducer with arbitrary directivity are mounted on one/two speakers; The local scattering and diffraction effects around the transducers result in complex directivity patterns. The directivity patterns can be obtained by analytical expressions, numerical simulations or measurements. 

In DEISM-ARG, we can model the room transfer function between transducers mounted on one/two speakers using the image source method while incorporating the local diffraction effects around the transducers. The local diffraction effects are captured using spherical-harmonic directivity coefficients obtained on a sphere around the transducers. In addition to DEISM in shoebox rooms, DEISM-ARG can model more complex room shapes. However, for version 2.0, we now only supports convex shapes. In short, DEISM-ARG has the following features: 

1. Arbitrary directivities of the source and receiver
2. Angle-dependent reflection coefficients
3. Convex room shapes

![image-20240812131054348](/docs/figures/scenario.png)

# Preparation and installing

## Build locally

- In you encounter **errors** like "unrecognized arguments: deism_envs_exact.yml", please type the following commands manuelly in the command line. 
- Clone or download the repository to your local directory
- Use the command `conda env create -f deism_env.yml` to create a Conda environment for the DEISM algorithms. If this does not work, try `conda env create -f deism_env_exact.yml` as the file **deism_envs_exact.yml** records the versions of all packages.
- Activate the created environment by "conda activate DEISM"
- Running `pip install -e .` will build the deism package including the c++ extensions locally. You can also modify the sources codes and check out the effects. In case you receiver errors like "ModuleNotFoundError: No module named 'pybind11" even after activated the conda environment, you can use `python -m pip install -e .` to try install again. 
- Run scripts in the **examples** folder. 

## Using pip to install remotely

- Run `pip install deism` to install deism. 

## Additional heads up
- You would need to an external LaTeX installation to render text properly when running the examples, since we enable `plt.rcParams["text.usetex"] = True` when plotting the simulated room transfer functions. Please refer to this page for more information [Matplotlib usetex Documentation](https://matplotlib.org/stable/users/explain/text/usetex.html). 

# Running codes

## Single set of parameters

The default parameters are defined in file **configSingleParam.yml** or **configSingleParam_ARG.yml**, depending on whether you want to run for shoebox rooms or more complicated room shapes. Take DEISM-ARG (**configSingleParam_ARG.yml**) as an example, there are two ways of running the codes: 

1. You can directly run **deism_arg_singleparam_example.py** in an IDE, which utilizes the parameters defined in **configSingleParam_arg.yml**.
2. You can run **deism_arg_singleparam_example.py** from the command line after activating the conda environment. In addition, you can access help information quickly by `python deism_arg_singleparam_example.py --help`. You can then change the parameters based on the instructions from the help message, e.g., `python deism_arg_singleparam_example.py -c 350 -zs 20` will change the parameter sound speed and the wall impedance. The new input value of the parameters then overrides the ones in file **configSingleParam_arg.yml**. After choosing the needed values, you can run the codes using, e.g., `python deism_arg_singleparam_example.py -c 350 -zs 20 --run`. 
3. You need to specify additionally the following parameters in the function `init_parameters` of **deism_arg_singleparam_example.py**:
   1. The vertices of the room
   2. If rotate the room w.r.t the origin. This can be useful if you want to have some comparisons with the rooms created using pyroomacoustics.
   3. The rotation angles of the room if it needs to be rotated.  
4. You can suppress all output information in the command line by adding flag "--quiet" or by setting the first parameters in the configuration.yml file SilentMode to 1. 



# Examples 

## DEISM-ARG

- An example of running DEISM-ARG is **examples/deism_arg_single_example.py**
  - You can run this from IDEs or via the command line, as introduced in the previous section
- An example of comparing different versions of DEISM-ARG is given in **examples/deism_args_compare.py**. The room transfer functions are compared among:
  - Original version (most computation-costly)
  - LC version (fastest)
  - Mix version (Trade-offs between Original and LC versions): Early reflections up to some changeable order (default is 2) are calculated using the original version and the higher orders are calculated using the LC version. 
- An example of comparing DEISM-ARG and pyroomacoustics is **examples/deism_arg_pra_compare.py**, Comparisons are done regarding if the following results are identical or mismatched only by a small deviation:
  - number of images
  - the positions of the images

## DEISM for shoebox

We provide two examples for running the original DEISM for shoebox rooms 

- An example of running DEISM-shoebox is **examples/deism_singleparam_example.py**
  - You can either run this from IDEs or via the cmd
- An example of comparing different DEISM versions are shown in **examples/deisms_lc_mix_test.py**. In this script, the following methods are compared
  - DEISM - original
  - DEISM - MIX (original + LC vectorized)
  - DEISM - LC vectorized 
  - FEM as groundtruth (only for this specific parameter settings)




## Tips

The example code only provides essential functionalities based on DEISM. For more complex scenarios, please contact the authors (zeyu.xu@audiolabs-erlangen.de) for support. In the following, a few important tips might be helpful for you: 

- If you want to simulate the scenario where both the source and receiver are positioned on the same speaker, you need to run DEISM for all reflection path except for the direct path. 
- It is recommended to set the distance at least 1m between the transducers and the walls. 



# Directivities 

Modeling the directivities of the source and receiver in the room acoustics simulation is receiving increasing attention. The directivities of the source or receiver can include both the transducer directional properties and the local diffraction and scatterring effects caused by the enclosure where the transducers are mounted. Modern smart speakers are typical embodiments of such scenarios. Human heads are also a very common case. 

## Simple directivities

- Monopole

## Arbitrary directivities

Some key information should be provided if you want to include your own directivity data:

1. Frequencies at which the directivities are simulated or measured. A 1D array. 
1. The spherical sampling directions around the transducer: azimuth from $0$ ( $+x$ direction) to $2 \pi$, inclination angle from $0$ ($+z$ direction)  to $\pi$. A 2D array with size (number of directions, 2).
1. The sampled pressure field at the specified directions and frequencies. A 2D array with size (number of frequencies, number of directions).
1. The radius of the sampling sphere. A 1D array or float number. 

For more information about directivity definition used in DEISM and DEISM-ARG, please refer to the following publication: 

> Zeyu Xu, Adrian Herzog, Alexander Lodermeyer, Emanuël A. P. Habets, Albert G. Prinn; Acoustic reciprocity in the spherical harmonic domain: A formulation for directional sources and receivers. JASA Express Lett. 1 December 2022; 2 (12): 124801. https://doi.org/10.1121/10.0016542






# Contributors 

- M. Sc. Zeyu Xu
- Songjiang Tan
- M. Sc. Hasan Nazım Biçer
- Dr. Albert Prinn
- Prof. Dr. ir. Emanuël Habets

 

# Academic publications

If you use this package in your research, please cite [our paper](https://doi.org/10.1121/10.0023935):

> Zeyu Xu, Adrian Herzog, Alexander Lodermeyer, Emanuël A. P. Habets, Albert G. Prinn; Simulating room transfer functions between transducers mounted on audio devices using a modified image source method. **J. Acoust. Soc. Am.** 1 January 2024; 155 (1): 343–357. https://doi.org/10.1121/10.0023935

> Z. Xu, E.A.P. Habets and A.G. Prinn; Simulating sound fields in rooms with arbitrary geometries using the diffraction-enhanced image source method, Proc. of International Workshop on Acoustic Signal Enhancement (IWAENC), 2024.



# Description of the codes and functions

## configSingleParam_ARG.yml

In this file you define the default parameters for DEISM-ARG to run. Note that this file is different from the  configSingleParam.yml on these parameters:

- The dimensions are defined separately in the example script. 
- You also need to specify if you want to rotate the room, and the rotation angles as well. 



## configSingleParam.yml

In this file you define the default parameters for DEISM-shoebox to run.



[^Euler]: https://mathworld.wolfram.com/EulerAngles.html