# Flask Server

## Contents

In this directory there are 4 main files:
- The flask server 'app.py' which holds the functionality to communicate with the Front-End
- The predictions script 'predictions.py' which runs the audio sent from the Front-End against the final LSTM model generated in the experimentation stage
- The unit testing suite 'test.py'
- The model 'lstm.h5' which holds the model used in prediction

## Running The Flask Server 

To run the server, the user must have Python 3.9 or greater installed. You can find the latest downloads [here](https://www.python.org/downloads/).

Steps to run are as follows:
1. Move to the root directory of the repository. If in the flaskServer directory use the command `cd ..` to move back to the root.
2. Install the correct python libraries. This is done by running on the commmand line `pip install -r requirements.txt`. This should install the relevant libraries.
3. Move back to the flaskServer directory using the command `cd flaskServer`
4. Type the command `flask run` to start the server

## Running the Flask Server To Communicate with the Front-End

1. Navigate to the 'stroke-app' directory to access the Front-End code
2. Run the command `expo start` to start the Expo project
3. In the terminal there should be an output similar to `> Metro waiting on exp://192.168.X.X:19000`. Copy the IP Address between `exp://` and `:19000` or whichever port number appears
4. Navigate back to the flaskServer Directory
5. Run the command `flask run -h 192.168.X.X` to hook the flask server to the Expo project

## Troubleshooting

If there are issues running the server, then open `requirements.txt` and try to install the libraries that may or may not be present in your python install. You can check the libraries kept on your version of python by running the command `pip freeze` on the command line.
