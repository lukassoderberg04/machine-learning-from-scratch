from enum import Enum
from pathlib import Path
from gui.app import App
from mnist.mnist_dataloader import MnistDataloader
from nn.cost import Cost
from nn.costs.mse import Mse
from nn.layer import Layer
from nn.memory import Memory
from nn.network import Network
from nn.networks.sequential import Sequential

class Mode(Enum):
    TRAIN = 0
    GUI   = 1

mainFilePath: Path = Path(__file__).resolve()
memory: Memory     = Memory()

# The sizes of the networks input / output.
INPUT_SIZE: int  = 28 * 28
OUTPUT_SIZE: int = 10

# General settings:
mode: Mode = Mode.TRAIN

# Training settings:
layers: list[Layer]                 = []
costFunction: Cost                  = Mse(OUTPUT_SIZE)
learningRate: float                 = 0.01
trainingNetwork: Network            = Sequential(layers, costFunction, learningRate)
trainingDataSetPath: Path           = mainFilePath / "mnist" / "data" / "mnist_test.csv"
batchSize: int                      = 10
trainingDataloader: MnistDataloader = MnistDataloader(trainingDataSetPath, batchSize)
epochsToTrain: int                  = 10
trainingNetworkSaveName: str        = "test_network"
trainingNetworkSavePath: Path       = mainFilePath

def Train():
    """
        Function for handling training.
    """
    for epoch in range(epochsToTrain):
        epochCost: float = trainingNetwork.TrainOneEpoch(trainingDataloader)
        print(f"Epoch {epoch} cost: {epochCost}")
        trainingDataloader.Reset()

    # Ask the user if you should save or not.
    answer = input("Save trained network? [y/n]: ").strip().lower()

    if answer in ("y", "yes"):
        memory.SaveNetwork(trainingNetwork, trainingNetworkSavePath)
        print("Network saved!")
    else:
        print("Network not saved!")

# GUI settings:
guiNetworkLoadPath: Path = mainFilePath / "test_network.pkl"
guiNetwork: Network      = memory.LoadNetwork(guiNetworkLoadPath)
app: App                 = App(guiNetwork)

def Gui():
    """
        Runs the application and the GUI.
    """
    app.Run()

def Main():
    """
        The Main entrypoint of the python program.
    """
    if mode == Mode.TRAIN:
        Train()
    elif mode == Mode.GUI:
        Gui()
    return

Main()