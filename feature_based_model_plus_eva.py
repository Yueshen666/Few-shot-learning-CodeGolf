import json
import random
from itertools import zip_longest
import numpy as np
from difflib import SequenceMatcher
import tokenize
import io
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import pandas as pd

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


def same_indicator(c1, c2):

    if ('print' in c1) and ('print' in c2):
        return 1
    if (('def' or 'lambda') in c1) and (('def' or 'lambda') in c2):
        return 1
    return 0


def get_all_keywords(c):
    try:
        s = [token[1] for token in tokenize.generate_tokens(
            io.StringIO(c).readline) if token[0] == 1]
    except:
        s = ['tokenize_error']
    return set(s)


def n_common_keywords(c1, c2):

    count = 0

    s1 = get_all_keywords(c1)

    s2 = get_all_keywords(c2)

    most_common = ['print', 'def', 'return', 'lambda', 'and', 'or', 'not',
                   'break', 'continue', 'True', 'False', 'if', 'else', 'for', 'while', 'in']

    common = list(s1.intersection(s2))
    for k in common:
        if k not in most_common:
            count += 1

    return count

# embarrassingly simple feature-based vectorizer


def vectorizer(tripl):
    # input: a triple containing input code 1, input code 2,
    # and a label y indicating if they are from the same topic/question
    c1, c2, y = tripl
    # input c1, c2 a code string pairs
    # outout a N feature vector

    # 1. absolute value of length difference. codes tackling the same question should have similar length
    abs_length_diff = np.abs(len(c1)-len(c2))
    # 2. length of longest common substring between c1 c2.
    match = SequenceMatcher(None, c1, c2).find_longest_match(0, len(c1), 0, len(c2))
    common_length = match[2]
    # 3. both has a print lambda or def?
    indicator = same_indicator(c1, c2)
    # 4. absolute value of the difference between the number of \n
    diff_newline = np.abs(c1.count("\n") - c2.count("\n"))
    # 5. absolute value of the difference between the number of ;
    diff_codesep = np.abs(c1.count(";") - c2.count(";"))
    # 6. number of common keyword overlap which are not common, sucn as 'ord'
    n_overlap = n_common_keywords(c1, c2)
    return [abs_length_diff, common_length, indicator, diff_newline, diff_codesep, n_overlap], y

# for 1 out of N eva


def get_sample(dic, n=6):
    keys = dic.keys()
    theN = random.sample(keys, n)
    theOneTrue = random.choice(theN)
    candi = []
    for k in theN:
        candi.append((k, random.choice(dic[k])))
    anchor = (theOneTrue, random.choice(dic[theOneTrue]))
    return {'anchor': anchor, 'candidates': candi}


def getScore(c1, c2):

    v, _ = vectorizer((c1, c2, -1))  # -1 = trivial
    return lr.predict_proba(np.array(v).reshape(1, -1))[0][1]


def getStats(idx):

    stats = {}
    tar = dataset[idx]['anchor'][1]
    # print(dataset[idx]['anchor'][0])
    for candi in dataset[idx]['candidates']:

        # print(candi[0],getScore(tar,candi[1]))
        stats[candi[0]] = getScore(tar, candi[1])

    return dataset[idx]['anchor'][0], {k: v for k, v in sorted(stats.items(), key=lambda item: item[1], reverse=True)}


def evaluate():
    matches = []
    for i in range(len(dataset)):
        tar, scores = getStats(i)
        pred = max(scores, key=scores.get)
        if tar == pred:
            matches.append(1)
        else:
            matches.append(0)
    return np.mean(matches)


if __name__ == "__main__":

    Siamese_full = create_pos_neg(restricted_data)
    print(len(Siamese_full))

    test1 = "x=input();print ord(x)"
    test2 = "print ord(input())"
    print('vectorizing:', test1, ' and ', test2)
    testx, _ = vectorizer((test1, test2, -1))
    print(testx)
    # make matrix data
    X = []
    y = []
    for tri in Siamese_full:
        x, label = vectorizer(tri)
        X.append(x)
        y.append(label)
    X = np.array(X)
    y = np.array(y)
    print(X.shape, y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    print('acc:', accuracy_score(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred))
    feature_name = ['abs_length_diff', 'common_length',
                    'same_indicator', 'diff_newline', 'diff_codesep', 'n_overlap']
    coeff = pd.DataFrame({'feature_name': feature_name,
                          'model_coefficient': lr.coef_.transpose().flatten()})
    # check coeffs:
    print(coeff)

    ###################################
    # 1 out of N classification evaluation N = 6

    held_out_data = {}
    for q, lst in data.items():
        restricted_lst = [s for s in lst if 100 < len(s) <= 200]
        if len(restricted_lst) >= 2:
            held_out_data[q] = restricted_lst

    held_out_total = len(held_out_data.items())
    print(held_out_total)  # held_out_data for 1 out of N test.
    oneFromSixTestSet = []  # do a 100 sample test each test for 1 out of N = 6
    for i in range(100):
        oneFromSixTestSet.append(get_sample(held_out_data))

    dataset = oneFromSixTestSet
    print()
    print('test for one case:')
    # change this number to any other number from 1-100 to see the result
    true_label, ranks = getStats(22)
    print('The true class is: ', true_label)
    print('Predicted:')
    for k, v in ranks.items():
        print(k, v)

    print('acc for 1 out of  N=6 test:', evaluate())  # about 80% acc
