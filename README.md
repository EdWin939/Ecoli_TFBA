# Predicting glycolytic flux is E. coli using TFBA

## Overview
This repository contains the code and experimental data for analyzing thermodynamic constarints impact on flux balance analysis of E. coli. 
This work was recently presented at RUG Honours College Closing symposium 2026.

## Repository Structure
Labwork/
├── Experiments/       # Original TIF files from the fluorescent microscopy, analysis scripts and a .pkl file output
├── Fiji/              # (Not tracked by Git) Place your ImageJ installation here
├── Dilution_calculator.xlsx   # Culture dilution spreadsheet
├── violinist.py      # Violin plot of fluorescence ratio distributions of the strains
├── violin_plot.png   # Premade violin plots
└── microbejSettings.xml  # Parameters used in ImageJ analysis
main_program/
├── examples/ecoli/textbook
    ├── datafiles/
        ├── allPhysioData_formatted.csv   # Original transport flux data from José Losa
        ├── metabolimics_Kochanowski.csv  # Original metabolite concentration data from Kochanowski et al. https://pmc.ncbi.nlm.nih.gov/articles/PMC5293157/
        ├── cleaned.scv                   # Cleaned versions of metabolite and flux data
        └── regressed_qm.npy  # Results of different regressions to avoid rerunning them
    ├── Ecoli_example_textbook.ipynb   # Original example notebook from `Thermo-Flux` ceators
    ├── final_ecoli_core_norange.ipynb # Main code fro builfing TFBA model, regression, prediction, and creating final figures
    └── methods_output.csv  # Saved final prediction values from all models. used for plotting without rerunning
├── final/ fluxes.csv    # All predicted fluxes on all conditions for variously constrained models. Saved for plotting
├── license and setup files
├── README.md  # Gurobi instalation instructions
└── thermo_flux/    # The oriignal `Thermo-Flux` package with minor but possibly necessary edits.


## Setup and Requirements
To run this project, you will need to set up a few dependencies that are not included in this repository due to file size limits:

1.  **Fiji (ImageJ):** You must have Fiji installed locally to process the image data.
2.  **SQLite Database:** The main database (`e_coli_core_compound.sqlite`) is not included. However, the code is designed to generate this database automatically upon the first run. 

## Usage
To replicate the analysis:
1. Clone this repository to your local machine.
2. Ensure you have the required Python dependencies installed.
3. Run the primary notebook: `final_ecoli_core_norange.ipynb`
