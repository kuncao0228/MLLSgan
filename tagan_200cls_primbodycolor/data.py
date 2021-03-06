import os
import numpy as np
from PIL import Image
import pickle
import nltk
from nltk.tokenize import RegexpTokenizer

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchfile
import pandas as pd

def split_sentence_into_words(sentence):
    tokenizer = RegexpTokenizer(r'\w+')
    return tokenizer.tokenize(sentence.lower())


def img_load_and_transform(img_path, img_transform=None):
    img = Image.open(img_path)
    if img_transform == None:
        img_transform = transforms.ToTensor()
    img = img_transform(img)
    if img.size(0) == 1:
        img = img.repeat(3, 1, 1)
    return img


class ReedICML2016(data.Dataset):
    def __init__(self):
        super(ReedICML2016, self).__init__()
        self.alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "
    
    def _get_word_vectors(self, desc, word_embedding, max_word_length):
        output = []
        len_desc = []
            #for i in range(desc.shape[1]):
            #words = self._nums2chars(desc[:, i])
            #print(words)
        
        words = split_sentence_into_words(desc)
            #print(words)
            #wd = self._nums2chars(desc[:, i+1])
            #print(wd)

        word_vecs = torch.Tensor([word_embedding.get_word_vector(w) for w in words])
            # zero padding
        if len(words) < max_word_length:
            word_vecs = torch.cat((
                                    word_vecs,
                                    torch.zeros(max_word_length - len(words), word_vecs.size(1))
                                    ))
        output.append(word_vecs)
        len_desc.append(len(words))
        #print(len_desc)
        return torch.stack(output), len_desc
    
    def _nums2chars(self, nums):
        chars = ''
        for num in nums:
            chars += self.alphabet[num - 1]
        return chars


class DatasetFromRAW(ReedICML2016):
    def __init__(self, img_root, caption_root, classes_fllename,
                 word_embedding, max_word_length, img_transform=None):
        super(DatasetFromRAW, self).__init__()
        self.max_word_length = max_word_length
        self.img_transform = img_transform
        
        self.data = self._load_dataset(img_root, caption_root, classes_fllename, word_embedding)
    
    def _load_dataset(self, img_root, caption_root, classes_filename, word_embedding):
        output = []
        with open(os.path.join(caption_root, classes_filename)) as f:
            lines = f.readlines()
            for line in lines:
                cls = line.replace('\n', '')
                filenames = os.listdir(os.path.join(caption_root, cls))
                for filename in filenames:
                    datum = torchfile.load(os.path.join(caption_root, cls, filename))
                    raw_desc = datum.char
                    desc, len_desc = self._get_word_vectors(raw_desc, word_embedding, self.max_word_length)
                    output.append({
                                  'img': os.path.join(img_root, datum.img), #adding / needed for running on drive
                                  'desc': desc,
                                  'len_desc': len_desc
                                  })
        return output
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        datum = self.data[index]
        img = img_load_and_transform(datum['img'], self.img_transform)
        desc = datum['desc']
        len_desc = datum['len_desc']
        # randomly select one sentence
        selected = np.random.choice(desc.size(0))
        desc = desc[selected, ...]
        len_desc = len_desc[selected]
        return img, desc, len_desc

def listToString(s):
    # initialize an empty string
    str1 = " "
    # return string
    return (str1.join(s))

class ConvertCapVec(ReedICML2016):
    def __init__(self):
        super(ConvertCapVec, self).__init__()

    def convert_and_save(self, caption_root, word_embedding, max_word_length):
        with open(os.path.join(caption_root, 'allclasses.txt'), 'r') as f:
            classes = f.readlines()
                #print(len(classes))
                #classes = '001.Black_footed_Albatross'
        with open('embed1029.pkl', 'rb') as file: # change with proper file name
            embed_data = pickle.load(file)
        for cls in classes:
            #cls = classes
            cls = cls[:-1]
            os.makedirs(caption_root + '_vec/' + cls)
            filenames = os.listdir(os.path.join(caption_root, cls))
            
            for filename in filenames:
                #print(filename)
                #print(cls)
                dict_key = filename
                dict_key = dict_key[:-2]
                dict_key = dict_key +'jpg'
                #print(dict_key)
                try:
                    newdesc = embed_data.get(dict_key)[0]
                except KeyError:
                    print('key error')
                    break
                #newdesc = listToString(newdesc)
                #print(newdesc)
                fj = os.path.join(caption_root, cls, filename)
                #print(fj)
                datum = torchfile.load(fj)
                raw_desc = datum.char
                # print(datum.img) -- bytes object
                #b'001.Black_footed_Albatross/Black_Footed_Albatross_0071_796113.jpg'
                desc, len_desc = self._get_word_vectors(newdesc, word_embedding, max_word_length)
                #print(desc.size())
                torch.save({'img': datum.img, 'word_vec': desc, 'len_desc': len_desc},
                        os.path.join(caption_root + '_vec', cls, filename[:-2] + 'pth'))


class ReadFromVec(data.Dataset):
    def __init__(self, img_root, caption_root, classes_filename, img_transform=None):
        super(ReadFromVec, self).__init__()
        self.img_transform = img_transform
        
        self.data = self._load_dataset(img_root, caption_root, classes_filename)
    
    def _load_dataset(self, img_root, caption_root, classes_filename):
        output = []
        with open(os.path.join(caption_root, classes_filename)) as f:
            lines = f.readlines()
            lines.pop(-1)
            for line in lines:
                cls = line.replace('\n', '')
                filenames = os.listdir(os.path.join(caption_root + '_vec', cls))
                #print(filenames)
                #print(len(filenames))
                #print(filenames)
                #print(len(filenames))
                for filename in filenames:
                    #print(cls)
                    #print(filename)
                    fname = os.path.join(caption_root + '_vec', cls, filename)
                    #print(fname)
                    datum = torch.load(fname)
                    output.append({
                                  'img': os.path.join(bytes(img_root, 'utf-8'), datum['img']), #adding / needed for running on drive
                                  'word_vec': datum['word_vec'],
                                  'len_desc': datum['len_desc']
                                  })
        return output
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        datum = self.data[index]
        img = img_load_and_transform(datum['img'], self.img_transform)
        word_vec = datum['word_vec']
        len_desc = datum['len_desc']
        # randomly select one sentence
        selected = np.random.choice(word_vec.size(0))
        word_vec = word_vec[selected, ...]
        len_desc = len_desc[selected]
        return img, word_vec, len_desc
