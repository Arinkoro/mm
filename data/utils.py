from PIL import Image
import os

def _get_img_from_path(img_path, transform=None):
    # if os.path.isfile(img_path) == True:
    with open(img_path, 'rb') as f:
        img = Image.open(f).convert('RGB')
    if transform is not None:
        img = transform(img)
    # else:
    #     with open('./data/image_retrieval/fashionIQ/toptee/B005S6XUUS.jpg', 'rb') as f:
    #         img = Image.open(f).convert('RGB')
    #     if transform is not None:
    #         img = transform(img)
        # print(img)
        # img = None
    return img
