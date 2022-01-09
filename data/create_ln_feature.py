from bert_serving.client import BertClient
import os
import json
import torch
import numpy as np
bc = BertClient()

_DEFAULT_FASHION_IQ_DATASET_ROOT = './image_retrieval/fashionIQ'


def get_bert_feature(dataset_root, clothing_type, split):
    data = os.path.join(dataset_root, 'captions/raw_cap',
                        'cap.{}.{}.json'.format(clothing_type, split))
    bert_data = []
    with open(data) as json_file:
        img_caption_data = json.load(json_file)

    for i in range(len(img_caption_data)):
        cap = ["The " + clothing_type + " " + img_caption_data[i]['captions'][0].replace(
            ".", "").lower() + " and " + img_caption_data[i]['captions'][1].lower()]
        cap = [ i for i in cap[0].split(" ") if i != ""]
        print(cap)
        bert_feature = bc.encode(cap)
        print(bert_feature.shape)
        np.save(os.path.join(dataset_root, 'captions/raw_cap/word_bert_data', split, clothing_type,
                'ln_bert_' + str(i + 1).zfill(5)), bert_feature)

    # data load
    # print(torch.Tensor(np.load(os.path.join(
    #     dataset_root, 'captions/raw_cap/tmp/ln_bert_00001.npy'))).shape)


if __name__ == '__main__':
    dataset_root = _DEFAULT_FASHION_IQ_DATASET_ROOT
    clothing_types = ['dress', 'toptee', 'shirt']
    splits = ['train', 'val']
    for cloth in clothing_types:
        for split in splits:
            get_bert_feature(dataset_root, cloth, split)
