# Deep Learning-Based Particle Detection and Instance Segmentation for Microscopy Images

Code to reproduce the instance segmentation results of my dissertation.

The tool *microbeSEG* with graphical user interface and OMERO support can be found [here](https://github.com/hip-satomi/microbeSEG).

## Installation
Clone the repository:
```
git clone https://github.com/TimScherr/DL_based_instance_segmentation_for_microscopy_images.git
```
Open the Anaconda Prompt (Windows) or the Terminal (Linux), go to the repository and create a new virtual environment:
```
cd path_to_the_cloned_repository
conda env create -f requirements.yml
```
Activate the virtual environment kit-sch-ge-2021_cell_segmentation_ve:
```
conda activate kit-sch-ge-2021_cell_segmentation_ve
``` 
Download all Cell Tracking Challenge Data (without Fluo-N3DL-DRO, Fluo-N3DL-TRIC, Fluo-N3DL-TRIF) and the evaluation software with
```
python download_data.py
```
About 40GiB free memory is needed. The training datasets with annotations are saved into *./train_data/* and the challenge data into *./challenge_data/*. In addition the [evaluation software](http://celltrackingchallenge.net/evaluation-methodology/) will be downloaded.

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
