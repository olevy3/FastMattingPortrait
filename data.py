from torch.utils.data import Dataset
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import resize
import numpy as np

# TODO make these configurable
DATAPATH = "./input"
GROUNDTRUTHPATH = "./training"

# TODO open the directories and check how many files aer present/what their
# names are instead of just hardcoding like this
TRAININGLEN = 10
TOTALLEN = 27

def padStr(s):
    return "0" * (2 - len(s)) + s

dimensions = (640, 480)

def getImg(name):
    return resize(io.imread(name), dimensions)

class DataCollection:
    def __init__(self, inputFilenames, truthFilenames):
        self.len = len(inputFilenames)
        self.inputFilenames = inputFilenames
        self.truthFilenames = truthFilenames

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        inputImg = getImg(self.inputFilenames[idx])
        truthImg = rgb2gray(getImg(self.truthFilenames[idx]))
        truthImg = np.reshape(truthImg, (640, 480, 1))
        return inputImg, truthImg, inputImg * truthImg

def get_training_set():
    trainingRange = range(1, TOTALLEN)
    return DataCollection(
        ["{}/GT{}.png".format(DATAPATH, padStr(str(i))) for i in trainingRange],
        ["{}/GT{}.png".format(GROUNDTRUTHPATH, padStr(str(i)))
            for i in trainingRange])

def get_test_set():
    testingRange = range(1, TRAININGLEN)
    return DataCollection(
        ["{}/GT{}.png".format(DATAPATH, padStr(str(i))) for i in testingRange],
        ["{}/GT{}.png".format(GROUNDTRUTHPATH, padStr(str(i)))
            for i in testingRange])
