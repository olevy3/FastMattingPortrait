import torch
import torch.nn as nn
import numpy as np
import sys

from torch.autograd import Variable
from skimage import io
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu

import bgs_model
import data

model = bgs_model.MattNet()
model.load_state_dict(torch.load("model_epoch_10.pth").state_dict())

inputImg = data.getImg(sys.argv[1])
print(inputImg.shape)
input_ = Variable(torch.tensor(np.reshape(inputImg, (1, 640, 480, 3))))
input_ = input_.float().permute(0, 3, 1, 2)

prediction = model(input_)
prediction = torch.cat((prediction, prediction, prediction), 1)
print(prediction.shape)
output = prediction.permute(0, 2, 3, 1)[0].data.numpy()
io.imsave("output.png", output)

grayscale = rgb2gray(output)
io.imsave("grayscale.png", grayscale)

thresh = threshold_otsu(grayscale)
thresholded = (grayscale >= (thresh * 0.95)).astype(np.float64)
io.imsave("thresh.png", thresholded)

product = output * inputImg
io.imsave("product.png", product)

# thresh = threshold_otsu(product)
productThresholded = product * np.reshape(thresholded, (640, 480, 1)) # (product >= thresh).astype(np.float64)
io.imsave("productT.png", productThresholded)

io.imsave("idk.png", product * np.reshape(rgb2gray(product), (640, 480, 1)))
# criterion = nn.MSELoss()
# expectedImg = data.getImg(sys.argv[1].replace("input", "training"))
# expected = Variable(torch.tensor(np.reshape(expectedImg, (1, 640, 480, 3))))
# expected = expected.float().permute(0, 3, 1, 2)
# mse = criterion(prediction, expected)
# 
# diff = (prediction - expected).permute(0, 2, 3, 1)[0].data.numpy()
# io.imsave("expected.png", expected.permute(0, 2, 3, 1)[0].data.numpy())
# io.imsave("diff.png", diff)
# print(mse.data.item())
