Introduction
============

Overview
--------

The Diffraction Enhanced Image Source Method - Arbitrary Room Geometry (DEISM-ARG) is a Python package that solves room acoustics problems involving source and receiver transducers with arbitrary directivity patterns. 

The package models the room transfer function between transducers mounted on one or two speakers using the image source method while incorporating local diffraction effects around the transducers.

Key Features
------------

DEISM-ARG provides the following capabilities:

1. **Arbitrary Directivities**: Support for arbitrary directivity patterns of both source and receiver transducers
2. **Angle-dependent Reflection**: Angle-dependent reflection coefficients with frequency- and wall-dependent impedance definition
3. **Complex Room Shapes**: Support for convex room shapes beyond simple shoebox geometries
4. **Local Diffraction Effects**: Incorporation of local scattering and diffraction effects captured using spherical-harmonic directivity coefficients

Problem Statement
-----------------

The code addresses scenarios where:

- Source and receiver transducers with arbitrary directivity are mounted on speakers
- Local scattering and diffraction effects around the transducers result in complex directivity patterns
- Directivity patterns can be obtained by analytical expressions, numerical simulations, or measurements
- Room transfer functions need to be modeled while accounting for these local effects

The local diffraction effects are captured using spherical-harmonic directivity coefficients obtained on a sphere around the transducers.

Applications
------------

DEISM-ARG is particularly useful for:

- **Smart Speaker Modeling**: Modern smart speakers with complex enclosures
- **Human Head Modeling**: Acoustic modeling involving human heads as scattering objects  
- **Custom Transducer Arrays**: Any scenario involving transducers mounted on complex geometries
- **Room Acoustics Research**: Academic and industrial research in room acoustics

Academic Background
-------------------

If you use this package in your research, please cite the following papers:

**Main Paper:**
    Zeyu Xu, Adrian Herzog, Alexander Lodermeyer, Emanuël A. P. Habets, Albert G. Prinn; 
    "Simulating room transfer functions between transducers mounted on audio devices using a modified image source method." 
    *J. Acoust. Soc. Am.* 155 (1): 343–357 (2024). 
    https://doi.org/10.1121/10.0023935

**Directivity Formulation:**
    Zeyu Xu, Adrian Herzog, Alexander Lodermeyer, Emanuël A. P. Habets, Albert G. Prinn; 
    "Acoustic reciprocity in the spherical harmonic domain: A formulation for directional sources and receivers." 
    *JASA Express Lett.* 2 (12): 124801 (2022). 
    https://doi.org/10.1121/10.0016542

**Arbitrary Geometries:**
    Z. Xu, E.A.P. Habets and A.G. Prinn; 
    "Simulating sound fields in rooms with arbitrary geometries using the diffraction-enhanced image source method," 
    *Proc. of International Workshop on Acoustic Signal Enhancement (IWAENC)*, 2024.

Contributors
------------

- M. Sc. Zeyu Xu
- Songjiang Tan  
- M. Sc. Hasan Nazım Biçer
- Dr. Albert Prinn
- Prof. Dr. ir. Emanuël Habets 