from enum import Enum
from pathlib import Path
from gui.app import App
from mnist.mnist_dataloader import MnistDataloader
from nn.cost import Cost
from nn.costs.mse import Mse
from nn.layer import Layer
from nn.layers.dense import Dense
from nn.layers.relu import Relu
from nn.layers.softmax import Softmax
from nn.memory import Memory
from nn.network import Network
from nn.networks.sequential import Sequential

class Mode(Enum):
    TRAIN = 0
    GUI   = 1
    TRAIN_THEN_GUI = 2

mainFilePath: Path = Path(__file__).resolve().parent
memory: Memory     = Memory()

# The sizes of the networks input / output.
INPUT_SIZE: int  = 28 * 28
OUTPUT_SIZE: int = 10

# General settings:
mode: Mode = Mode.TRAIN_THEN_GUI

# Training settings:
layers: list[Layer]                 = [
    Dense(INPUT_SIZE, 16),
    Relu(16),
    Dense(16, OUTPUT_SIZE),
    Relu(OUTPUT_SIZE),
    Softmax(OUTPUT_SIZE)
]
costFunction: Cost                    = Mse(OUTPUT_SIZE)
learningRate: float                   = 0.01
trainingNetwork: Network              = Sequential(layers, costFunction, learningRate)
trainingDataSetPath: Path             = mainFilePath / "mnist" / "data" / "mnist_train.csv"
batchSize: int                        = 10
trainingDataloader: MnistDataloader   = MnistDataloader(trainingDataSetPath, batchSize)
epochsToTrain: int                    = 10
trainingNetworkSaveName: str          = "first_run.pkl"
trainingNetworkSavePath: Path         = mainFilePath / trainingNetworkSaveName
evaluationDataSetPath: Path           = mainFilePath / "mnist" / "data" / "mnist_test.csv"
evaluationDataloader: MnistDataloader = MnistDataloader(evaluationDataSetPath, batchSize)

def Train() -> Network:
    """
        Function for handling training.
    """
    for epoch in range(epochsToTrain):
        epochCost: float = trainingNetwork.TrainOneEpoch(trainingDataloader)
        epochCost = epochCost / learningRate # Since learning rate has already influenced the cost.
        print(f"Epoch {epoch + 1} cost: {epochCost}")
        trainingDataloader.Reset()

    accuracy: float = trainingNetwork.Evaluate(evaluationDataloader)

    print(f"Accuracy of trained model: {accuracy}.")

    # Ask the user if you should save or not.
    answer: str = input("Save trained network? [y/n]: ").strip().lower()

    if answer in ("y", "yes"):
        memory.SaveNetwork(trainingNetwork, trainingNetworkSavePath)
        print("Network saved!")
    else:
        print("Network not saved!")

    return trainingNetwork

# GUI settings:
guiNetworkLoadPath: Path = mainFilePath / "first_run.pkl"

def Gui(network: Network | None = None):
    """
        Runs the application and the GUI.
    """
    if network is None:
        guiNetwork: Network      = memory.LoadNetwork(guiNetworkLoadPath)
        app: App                 = App(guiNetwork)
    else:
        guiNetwork: Network = network
        app: App            = App(guiNetwork)

    app.Run()

def Main():
    """
        The Main entrypoint of the python program.
    """
    if mode == Mode.TRAIN:
        Train()
    elif mode == Mode.GUI:
        Gui()
    elif mode == Mode.TRAIN_THEN_GUI:
        network: Network = Train()
        Gui(network)

    return

Main()