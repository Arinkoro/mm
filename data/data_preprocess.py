# utf-8
"""
This code is to create the new caption dataset
without missing data
"""
import json
import os

_DEFAULT_FASHION_IQ_DATASET_ROOT = './image_retrieval/fashionIQ'
_DEFAULT_FASHION_IQ_VOCAB_PATH = './image_retrieval/fashionIQ/fashion_iq_vocab.pkl'


def get_img_caption_json(dataset_root, clothing_type, split):
    with open(os.path.join(dataset_root, 'captions', 'cap.{}.{}.json'.format(clothing_type, split))) as json_file:
        img_caption_data = json.load(json_file)
    new_data = []
    for i in range(len(img_caption_data)):
        target_path = os.path.join(
            dataset_root, "all_imgs/" + clothing_type + '/' + img_caption_data[i]["target"] + ".jpg")
        ref_path = os.path.join(
            dataset_root, "all_imgs/" + clothing_type + '/' + img_caption_data[i]["candidate"] + ".jpg")
        if os.path.isfile(target_path) == True and os.path.isfile(ref_path) == True:
            new_data.append(img_caption_data[i])

    with open(os.path.join(dataset_root, 'captions/raw_cap/', 'cap.{}.{}.json'.format(clothing_type, split)), 'a') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    dataset_root = _DEFAULT_FASHION_IQ_DATASET_ROOT
    clothing_types = ['dress', 'toptee', 'shirt']
    split = 'val'
    for clothing_type in clothing_types:
        get_img_caption_json(dataset_root, clothing_type, split)
