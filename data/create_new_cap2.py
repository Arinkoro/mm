# utf-8
"""
This code is to create the new caption dataset
without missing data
"""
import json
import os
import cv2
import numpy as np

_DEFAULT_FASHION_IQ_DATASET_ROOT = './image_retrieval/fashionIQ'
_DEFAULT_FASHION_IQ_VOCAB_PATH = './image_retrieval/fashionIQ/fashion_iq_vocab.pkl'


def get_img_caption_json(dataset_root, clothing_type, split):
    with open(os.path.join(dataset_root, 'captions/raw_cap/new_cap', 'cap.{}.{}.json'.format(clothing_type, split))) as json_file:
        img_caption_data = json.load(json_file)
    new_data = []
    for i in range(len(img_caption_data)):
        target_path = os.path.join(
            dataset_root, "all_imgs", clothing_type, "U-2Net", img_caption_data[i]["target"] + ".jpg")
        if os.path.isfile(target_path) == True:
            img = cv2.imread(target_path)
        # im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # if np.median(img) == 0.0:
            #     img_caption_data[i]['target_mid'] = np.mean(img)
            # else:
            cap = img_caption_data[i]["captions"][0] + " " + img_caption_data[i]["captions"][1]
            if "black" in cap:
                img_caption_data[i]['target_mid'] = np.median(img)
            else:
                img_caption_data[i]['target_mid'] = np.median(img[img!=0])
            # print(img_caption_data[i])
        # if os.path.isfile(target_path) == True and os.path.isfile(ref_path) == True:
            new_data.append(img_caption_data[i])
        else:
            print(target_path)

    with open(os.path.join(dataset_root, 'captions/raw_cap/new_cap/u2netver', 'cap.{}.{}.json'.format(clothing_type, split)), 'a') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    dataset_root = _DEFAULT_FASHION_IQ_DATASET_ROOT
    clothing_types = ['toptee']
    split = 'val'
    for clothing_type in clothing_types:
        get_img_caption_json(dataset_root, clothing_type, split)
