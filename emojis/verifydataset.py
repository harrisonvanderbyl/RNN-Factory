import torch

data = torch.load('./dataset.pt')

print(data[0].shape)

from PIL import Image,GifImagePlugin
import torchvision.transforms as transforms
data = [d[0:1,:,:,:] for d in data]
outputTensor = torch.cat(data, dim=0)/255.0

# fill alpha channel with black
GifImagePlugin.GifImageFile.save(transforms.ToPILImage()(outputTensor[0].cpu()),'./verify.gif', save_all=True, append_images=[
        transforms.ToPILImage()(outputTensor[i].cpu()) for i in range(1,outputTensor.shape[0])
    ], loop=0, frame_duration=5, transparency=0)