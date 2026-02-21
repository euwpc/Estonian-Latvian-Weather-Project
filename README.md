# SWP – Baltic Weather Project

Visualize meteorological data using high-quality maps generated from NetCDF files.  
Built with Python, Matplotlib, Cartopy, NumPy, and SciPy.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python version">
  <img src="https://img.shields.io/badge/Dependencies-matplotlib%20%7C%20cartopy%20%7C%20netCDF4-success?style=for-the-badge" alt="Dependencies">
  <img src="https://img.shields.io/github/repo-size/euwpc/SWP?style=for-the-badge&color=informational" alt="Repo size">
  <img src="https://img.shields.io/github/last-commit/euwpc/SWP?style=for-the-badge&color=important" alt="Last commit">
</p>

---

## Features

- High-resolution geographic mapping with Cartopy  
- Support for contour plots, filled contours, wind barbs, quiver vectors, and more  
- Compatible with common meteorological variables (temperature, pressure, wind, precipitation, etc.)  
- Efficient handling of large NetCDF datasets  
- Simple command-line execution  

---

## Quick Start

### 1. Install Anaconda Prompt

Download from: https://www.anaconda.com/download

### 2. Open Anaconda Prompt

Search in Windows for:

Anaconda Prompt

Open it.

### 3. Create a Clean Environment

Use Python 3.11 for best compatibility:

`conda create -n baltic_env python=3.11`

Press `y` to confirm.

### 4. Activate the Environment

`conda activate baltic_env`

You should now see:

`(baltic_env)`

at the beginning of the line.


### 5. Install Required Packages
Install everything from conda-forge (precompiled, no build errors):

`conda install -c conda-forge cartopy matplotlib numpy scipy shapely pillow requests netcdf4`

Wait until installation completes.

Do not use pip for Cartopy.

### 6. Install Git (If Needed)

Check if Git is installed:

`git --version`

If not recognized, download Git:

https://git-scm.com/download/win

During installation choose:
“Git from the command line and also from 3rd-party software”

Restart Anaconda Prompt after installing.

### 7. Clone the Repository

Inside Anaconda Prompt:

`git clone https://github.com/euwpc/Baltic-Weather-Project.git`
`cd Baltic-Weather-Project`

### 8. Run the Program

To run the program, run: `python main.py`

### Output 

Generated PNG maps will appear inside the project folder.

To open the folder quickly:

`explorer .` 

(You will need to stop auto-updating in order to open the folder (CTRL + C to stop auto-updating))

### Running the Project Later

Open Anaconda Prompt and run:

`conda activate baltic_env`
`cd C:\Users\YOURNAME\Baltic-Weather-Project`
`python main.py`

### Future updates

I will continue working on this project, and once I'm done with an update, you need to run `git pull` in Anaconda Prompt to install the new update.
