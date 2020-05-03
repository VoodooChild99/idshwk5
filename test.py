# domain_name_len, num_of_diff_letter, num_of_diff_num, entropy
import math, os
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def cal_diff_letter_num(s):
    letter_list = []
    number_list = []
    for letter in s:
        if letter.isalpha():
            if letter not in letter_list:
                letter_list.append(letter)
        elif letter.isdigit():
            if letter not in number_list:
                number_list.append(letter)
    return len(letter_list), len(number_list)

def cal_string_entropy(s):
    letter_dict = {}
    for letter in s:
        if letter in letter_dict:
            letter_dict[letter] += 1
        else:
            letter_dict[letter] = 1
    string_length = len(s)
    entropy = 0.0
    for (letter, cnt) in letter_dict.items():
        p = 1.0 * cnt / string_length
        entropy += -(p * math.log2(p))
    return entropy

# Read training dataset from file
def read_and_preprocess_training_data(path):
    raw_data = []
    labels = []
    with open(path, 'r') as f:
        for line in f.readlines():
            splited = line.strip().split(',')
            tup = [0 , 0, 0, 0]
            tup[0] = len(splited[0])
            tup[1], tup[2] = cal_diff_letter_num(splited[0])
            tup[3] = cal_string_entropy(splited[0])
            tup = np.array(tup)
            raw_data.append(tup)
            if splited[1] == 'dga':
                labels.append(1)
            else:
                labels.append(0)
    raw_data = np.array(raw_data)
    labels = np.array(labels)
    return raw_data, labels


def read_and_preprocess_test_data(path):
    raw_data = []
    orig_domain = []
    with open(path, 'r') as f:
        for line in f.readlines():
            splited = line.strip()
            orig_domain.append(splited)
            tup = [0, 0, 0, 0]
            tup[0] = len(splited)
            tup[1], tup[2] = cal_diff_letter_num(splited)
            tup[3] = cal_string_entropy(splited)
            tup = np.array(tup)
            raw_data.append(tup)
    raw_data = np.array(raw_data)
    return raw_data, orig_domain


training_data = read_and_preprocess_training_data('./train.txt')
clf = RandomForestClassifier(random_state=0)
clf.fit(training_data[0], training_data[1])
test_data, orig_domain = read_and_preprocess_test_data('./test.txt')

predict_result = clf.predict(test_data)

with open('./result.txt', 'w') as f:
    num_item = len(orig_domain)
    for i in range(num_item - 1):
        if predict_result[i] == 0:
            f.write('{0},nodga\n'.format(orig_domain[i]))
        else:
            f.write('{0},dga\n'.format(orig_domain[i]))
    if predict_result[num_item - 1] == 0:
        f.write('{0},nodga'.format(orig_domain[num_item - 1]))
    else:
        f.write('{0},dga'.format(orig_domain[num_item - 1]))