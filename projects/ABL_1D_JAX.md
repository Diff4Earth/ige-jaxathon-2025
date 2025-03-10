## 👩‍🏫 People involved : 
- Ruth Price
- Aurélie Albert
- Charlotte Durand
- Sammy Metref (friend)

## 🧊 Project description 
The idea is to evaluate the work that has to be done in order to translate a one-dimensional ABL model from Fortran (NEMO, SI3) into JAX. 
Two different approaches:
- Extracting code from NEMO/SI3/ABL (Lemarié's code, public repo)
- Extracting code from NANUK/ABL (WIP, unpublished, unofficial code) - NANUK is already an extraction from NEMO (80% of the job done yay!)

The idea afterward would be to couple this ABL with a sea ice emulator, which means we also need to list all the necessary variables.

## ❄️ Background information : 
- Reference paper: A simplified atmospheric boundary layer model for an improved representation of air–sea interactions in eddying oceanic models: implementation and first evaluation in NEMO (4.0),, Lemarié et al., https://doi.org/10.5194/gmd-14-543-2021
- Reference code: https://zenodo.org/records/3904518
- [Implementation in NEMO](https://forge.nemo-ocean.eu/nemo/nemo/-/tree/main/src/ABL?ref_type=heads)

## 💻 Planned work : 
1. Looking at all the routines, evaluate the dependencies
2. Look into different tools for the translation (LLM?)
3. Select one or several (max 3) small routines to start the translation into JAX
4. Define a test case to test the routines


## ⛳️ Success metrics : 
- Roadmap to translate the ABL code in JAX. (List of routines)
- If possible, JAX routine for one or two functions
