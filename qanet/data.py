import os
import json
import numpy as np
import urllib.request
import zipfile
import spacy

data_dir = "data"

class SQuAD:
    def __init__(self):
        self.maybedownload()
        with open(os.path.join(data_dir, 'train-v1.1.json')) as f:
            self.train = json.load(f)
        with open(os.path.join(data_dir, 'dev-v1.1.json')) as f:
            self.dev = json.load(f)
        # self.glove = self.load_glove()
        self.embed()

    def load_glove(self):
        print("Loading GloVe Model")
        f = open(os.path.join(data_dir, 'glove.6B.300d.txt'),'r')
        model = {}
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word] = embedding
        print("Done.",len(model)," words loaded!")
        return model

    def maybedownload(self):
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        if not os.path.exists(os.path.join(data_dir, 'train-v1.1.json')):
            print("Downloading Squad")
            response = urllib.request.urlopen('https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json')
            train_json = response.read()
            response = urllib.request.urlopen('https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json')
            dev_json = response.read()
            with open(os.path.join(data_dir, 'train-v1.1.json'), 'w') as f:
                f.write(train_json)
            with open(os.path.join(data_dir, 'dev-v1.1.json'), 'w') as f:
                f.write(dev_json)
        if not os.path.exists(os.path.join(data_dir, 'glove.6B.zip')):
            print("Downloading GloVe")
            response = urllib.request.urlopen('http://nlp.stanford.edu/data/glove.6B.zip')
            glove_zip = response.read()
            with open(os.path.join(data_dir, 'glove.6B.zip'), 'wb') as f:
                f.write(glove_zip)
            zip_ref = zipfile.ZipFile(os.path.join(data_dir, 'glove.6B.zip'), 'r')
            zip_ref.extractall(data_dir)
            zip_ref.close()

    def embed(self):
        self.embed_data(self.train)

    def embed_data(self, data):
        nlp = spacy.load('en')
        data_list = data['data']
        for data_elem in data_list:
            paragraphs = data_elem['paragraphs']
            for paragraph in paragraphs:
                context = paragraph['context']
                doc = nlp(context)
                print([(w.text, w.pos_) for w in doc])
                qas = paragraph['qas']
                for qa in qas:
                    answers = qa['answers']
                    for ans in answers:
                        start = ans['answer_start']
                        end = start + len(ans['text'])
                        sys.exit(0)

if __name__ == '__main__':
    squad = SQuAD()
