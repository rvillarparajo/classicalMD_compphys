# Project 1: Molecular dynamics simulation of Argon atoms
> In this project, we build a system to simulate up to 108 Argon atoms in a lattice structure in 3D model, and implement some useful functions to investigate physical observables in the system.

## Table of Contents
* [General Info](#general-information)
* [Features](#features)
* [Setup](#setup)
* [General Structure](#General-structure)
* [Project Status](#project-status)
* [Acknowledgements](#acknowledgements)
* [Contact](#contact)
* [Development Roadmap](#development-roadmap)


## General Information
- This project is mainly aimed at exploring code-simulated behavior of a system to facilitate the study of physics theory behind it.
- The purpose of our project is using code to simulate molecular interactions. by changing parameters, we can easily find out how the system behave according to different temperature and so on.
- The code also produce clear plots so that we can compare our theoretically simulated data with experimental data.


## Features
List the ready features here:
- Simulates the Argon molecular system in the 3d box
- Plots energy system with conservation of total energy
- Animate the motion of the particles in a 3d box
- Calculates and plots selected observables (pressure and specific heat) for investigation with various methods
- Analyze the data


## Setup
- Run the first and the second cell to import everything we need.
- To display a 3D animation, we may need to install ffmpeg in the PATH. If an error shows "ffmpeg not found", then please refer to this [tutorial](https://phoenixnap.com/kb/ffmpeg-windows) for troubleshoting.
- To display a 3D animation, we have to set the 'animation' to True and the animation is displayed using HTML from IPython.display.
- To apply the rescailing, we have to set 'rescale' to True.
- To dispay the energy plots, we have to set 'energy_plot' to True.


## General Structure

The **global variables** are defined and described in detail in file _global_constants.py_
For simulation purpose, we can change T (unitless temperature) and $\rho$ (density).

The **functions** are stored and described in detail in file _utilities.py_.

The **execution** of our system can be achieved in file _main.py_ or preferably _main.pynb_. In Jupyter Notebook version, it is possible to run code cell by cell and get clear result. We first import variables and functions and then operate our simulation system. Also it is possible to run the code for the plots that are included in the report. 


#### Extra:Data analysis and  Correctness checks
We set up some plots that analyse our data of our experiment, as well as functions to test the correctness of our underlying logic of the code.



## Project Status
Project is:  _complete_ 


## Acknowledgements
Give credit here.
- This project was designed by TU Delft Computational Physics course, based on [this description](https://compphys.quantumtinkerer.tudelft.nl/proj1-moldyn-description/).


## Contact
Created by [@csfetsou](https://gitlab.kwant-project.org/csfetsou)[@rvillarparajo](https://gitlab.kwant-project.org/rvillarparajo)[@kimmyzhao](https://gitlab.kwant-project.org/kimmyzhao)


## Development Roadmap
Here we followed this roadmap to implement our method. This list is fully checked and reviewed.

