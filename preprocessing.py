# this file convert code text to tensors using pretrained codeBERT.
import json
import random
from itertools import zip_longest
import torch
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel
import numpy as np

# load data
with open('python_golfs.json', 'r') as f:
    data = json.load(f)
restricted_data = {}
for q, lst in data.items():
    restricted_lst = [s for s in lst if len(s) <= 100]  # get rid of too long hack solutions
    if len(restricted_lst) >= 2:
        restricted_data[q] = restricted_lst

total = len(restricted_data.items())
print(total)  # 1420 unique categories/topics


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaModel.from_pretrained("microsoft/codebert-base")
# model.to(device)


def code2vec(c, debugg=False):
    code_tokens = tokenizer.tokenize(c)
    tokens = [tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
    context_embeddings = model(torch.tensor(tokens_ids)[None, :])[0]
    if debugg:
        print(f"tokens:{tokens}\n Dims: {context_embeddings.shape}")
    return context_embeddings[0][0]  # only use the [CLS] repr for the code string


def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)


def create_pos_neg(dic):

    # each item contains either (c1,c2,1) (positive) or ( c1, c2, 0) negative - not belongs to same question
    Entries = []
    for q, lst in dic.items():

        # create positive pairs for each question q:
        temp = lst.copy()
        random.shuffle(temp)
        for c1, c2 in grouper(temp, 2):
            if c2:
                Entries.append((c1, c2, 1))
        for anchor in temp:
            # create negative pair
            # first select a key (not != q)
            negk = random.choice([key for key in list(dic) if key != q])
            # random select an example of that key (class)

            neg_example = random.choice(dic[negk])
            Entries.append((anchor, neg_example, 0))
    return Entries


if __name__ == "__main__":

    # each item contain (two code string x1 and x2 and a label indicating
    # if x1 x2 are in for the same question)

    Siamese_full = create_pos_neg(restricted_data)
    print(len(Siamese_full))

    print(np.mean([tri[2] for tri in Siamese_full]))  # pos : neg is about 30% : 70%

    # save all labels line by line to file
    with open('y.txt', 'w') as f:
        for idx, tri in enumerate(Siamese_full):
            if idx % 1000 == 0:
                print(f'working on the {idx} entry')
            f.write("%s\n" % tri[2])

    # save x1's bert CLS representation line by line each line is a 768, array
    with open('x1.txt', 'ab') as f:
        for idx, tri in enumerate(Siamese_full):
            if idx % 1000 == 0:
                print(f'working on the {idx} entry')
            arr = code2vec(tri[0]).detach().numpy()
            np.savetxt(f, [arr], delimiter=',')
            # f.write(b"\n")

    # save x2's bert CLS representation line by line each line is a 768, array
    with open('x2.txt', 'ab') as f:
        for idx, tri in enumerate(Siamese_full):
            if idx % 1000 == 0:
                print(f'working on the {idx} entry')
            arr = code2vec(tri[1]).detach().numpy()
            np.savetxt(f, [arr], delimiter=',')
