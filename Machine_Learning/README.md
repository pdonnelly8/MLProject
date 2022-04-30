# Machine Learning (ML) Directory

## Contents

In this directory there are:
- Scripts to run the CNN model and the tone classification
- Scripts to run the LSTM model and the tone classification
- Statistics that have been used within the Dissertation

## Running the ML Pipelines

To run the pipelines, the user must have Python 3.9 or greater installed. You can find the latest downloads [here](https://www.python.org/downloads/).

Steps to run are as follows: 
1. Move to the root directory of the repository. If in the Machine_Learning directory use the command `cd ..` to move back to the root.
2. Install the correct python libraries. This is done by running on the commmand line `pip install -r requirements.txt`. This should install the relevant libraries.
3. Move back to the Machine_Learning directory using the command `cd Machine_Learning`
2. On the command line run `python3 -m (pipeline_name).py`. This should then execute the the pipelines on the command line.

## Accessing the Statistics.

The directory `ExperimentationStats` holds the statistics gathered during experimentation. The filing hierarchy is the following:

- "'ML\_Approach'/'Learning\_Rate'/'Batch\_Size'"

Inside these are screenshots of the Loss values, Accuracy Values, and Confusion Matrices that were produced at those specific parameters. As well as this, at the 'ML\_Approach' level there are also spreadsheets containing the metrics that were used from tables 3-12 in the dissertation.

## Troubleshooting

If there are issues running the pipeline, then open `requirements.txt` and try to install the libraries that may or may not be present in your python install. You can check the libraries kept on your version of python by running the command `pip freeze` on the command line.