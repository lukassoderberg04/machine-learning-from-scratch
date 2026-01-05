# Machine learning from scratch
A personal project made to learn more about how machine learning is implemented in practice.

## Running the project

### Prerequisites
This project depends on the python library **Numpy**.

### Executing tests
By using pythons built in testing module, you can run all tests from the root directory of this project. Do this by typing this into the command line with the working directory set to the project root:

```bash
python -m unittest
```

This is a good way to make sure that some of the projects functionality works as intended.

### Running the project
To test the project, run python main.py, but first make sure to change the settings in the file! Especially the mode of the program! Ex:

```python
# General settings:
mode: Mode = Mode.GUI
```

## Project structure
* **/mnist/**: Directory containing helper classes and utility functions for loading the mnist dataset, used in training and validating the different models.

* **/mnist/data/**: Directory that contains the .csv files used in training the model. The format of these are: value[0] := the classified image and value[1:] := the pixel grayscale data.

* **/mnist/tests/**: Directory containing tests for the mnist library.

* **/nn/**: Directory containing classes and other function for creating and defining a neural network.

* **nn/tests/**: Directory for tests regarding neural network.

* **nn/layers/**: Directory of all layers that could be used to setup a network.

* **nn/networks/**: Directory of base objets for storing the layers and executing the training and evaluation.

* **nn/costs/**: Directory of the cost layers responsible for computing the cost function.

* **gui/**: Directory containing code for the graphical interface.