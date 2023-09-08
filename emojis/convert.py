# Import the required libraries
import torch
from PIL import Image,GifImagePlugin
import torchvision.transforms as transforms

# Read urls.txt
with open('./urls.txt') as f:
    urls = f.readlines()

data = []
number = 0
for url in urls:
    # download file using python
    url = "https://emoji.discadia.com/emojis/resized/" + url.strip()

    # download file using python
    import urllib3
    http = urllib3.PoolManager()
    r = http.request('GET', url)
    if "NoSuchKey" in str(r.data):
        continue
    with open('./test.webp', 'wb') as f:
        f.write(r.data)



    # run mogrify to convert to gif
    import os
    os.system("mogrify -resize 48x48! -format gif *.webp")

    # delete the webp file
    os.system("rm *.webp")

    

    # Read the image

    image = GifImagePlugin.GifImageFile('./test.gif')

    
    outputTensor = torch.zeros(image.n_frames, 3, image.size[1], image.size[0], dtype=torch.uint8)
    print (outputTensor.shape)
    for i in range(image.n_frames):
        image.seek(i)
        outputTensor[i] = (transforms.ToTensor()(image.convert('RGB'))*255).to(torch.uint8)

    
    data += [outputTensor]

    number += 1

    if number == 100:
        break

torch.save(data, './dataset.pt')
    

    # print the image
    # image.seek(0)
    # GifImagePlugin.GifImageFile.save(transforms.ToPILImage()(outputTensor[0].cpu()),'./test2.gif', save_all=True, append_images=[
    #     transforms.ToPILImage()(outputTensor[i].cpu()) for i in range(1,outputTensor.shape[0])
    # ], loop=0, frame_duration=5)