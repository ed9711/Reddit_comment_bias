import numpy as np
import argparse
import json
import string
import csv
import re
import datetime

start_time = datetime.datetime.now()
# Provided wordlists.
FIRST_PERSON_PRONOUNS = {
    'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'}
SECOND_PERSON_PRONOUNS = {
    'you', 'your', 'yours', 'u', 'ur', 'urs'}
THIRD_PERSON_PRONOUNS = {
    'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them',
    'their', 'theirs'}
SLANG = {
    'smh', 'fwb', 'lmfao', 'lmao', 'lms', 'tbh', 'rofl', 'wtf', 'bff',
    'wyd', 'lylc', 'brb', 'atm', 'imao', 'sml', 'btw', 'bw', 'imho', 'fyi',
    'ppl', 'sob', 'ttyl', 'imo', 'ltr', 'thx', 'kk', 'omg', 'omfg', 'ttys',
    'afn', 'bbs', 'cya', 'ez', 'f2f', 'gtr', 'ic', 'jk', 'k', 'ly', 'ya',
    'nm', 'np', 'plz', 'ru', 'so', 'tc', 'tmi', 'ym', 'ur', 'u', 'sol', 'fml'}

path = "../feats/"
# path = "/u/cs401/A1/feats/"
alt_feats = np.load(path + "Alt_feats.dat.npy")
center_feats = np.load(path + "Center_feats.dat.npy")
left_feats = np.load(path + "Left_feats.dat.npy")
right_feats = np.load(path + "Right_feats.dat.npy")


def read_file(path):
    with open(path, 'r') as f:
        lines = f.read().splitlines()
    return lines


alt_IDs = read_file(path + "/Alt_IDs.txt")
center_IDs = read_file(path + "/Center_IDs.txt")
left_IDs = read_file(path + "/Left_IDs.txt")
right_IDs = read_file(path + "/Right_IDs.txt")

path2 = "../Wordlists/"
# path2 = "/u/cs401/Wordlists/"
norms = open(path2 + "BristolNorms+GilhoolyLogie.csv", 'r')
norms = csv.reader(norms)
AoA = {}
IMG = {}
FAM = {}
first_line = True
for line in norms:
    if first_line:
        first_line = False
        continue
    AoA[line[1]] = line[3]
    IMG[line[1]] = line[4]
    FAM[line[1]] = line[5]
norms = open(path2 + "Ratings_Warriner_et_al.csv", 'r')
norms = csv.reader(norms)
V = {}
A = {}
D = {}
first_line = True
for line in norms:
    if first_line:
        first_line = False
        continue
    V[line[1]] = line[2]
    A[line[1]] = line[5]
    D[line[1]] = line[8]


def extract1(comment):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''
    # TODO: Extract features that rely on capitalization.
    # TODO: Lowercase the text in comment. Be careful not to lowercase the tags. (e.g. "Dog/NN" -> "dog/NN").
    # TODO: Extract features that do not rely on capitalization.

    feats = np.zeros((1, 29))

    plus_tags = re.compile("([\w]+|[\W]+|[\/])/([\w]+[\s]|[\W]+[\s])").findall(comment)
    words = re.compile("([\w]+|[\W]+|[\/])/(?=[\w]+[\s]|[\W]+[\s])").findall(comment)
    l = len(plus_tags)
    tags = []
    for i in range(l):
        tags.append(plus_tags[i][1].rstrip(' '))
    # 2
    # first_person = ('i', 'me', 'we', 'us', 'my', 'mine', 'our', 'ours')
    # disregard case
    # if plus_tags[i][0].lower() in FIRST_PERSON_PRONOUNS:
    feats[0, 1] = len(re.compile(r'\b(' + r'|'.join(FIRST_PERSON_PRONOUNS) + r')\b').findall(comment))

    # 3
    # second_person = ('you', 'your', 'yours', 'u', 'ur', 'urs')
    # disregard case
    # if plus_tags[i][0].lower() in SECOND_PERSON_PRONOUNS:
    feats[0, 2] = len(re.compile(r'\b(' + r'|'.join(SECOND_PERSON_PRONOUNS) + r')\b').findall(comment))

    # 4
    # third_person = ('he', 'she', 'it', 'him', 'her', 'his', 'hers', 'its', 'they', 'them', 'their', 'theirs')
    # disregard case
    # if plus_tags[i][0].lower() in THIRD_PERSON_PRONOUNS:
    feats[0, 3] = len(re.compile(r'\b(' + r'|'.join(THIRD_PERSON_PRONOUNS) + r')\b').findall(comment))

    # 5
    # if plus_tags[i][1].rstrip(' ') == 'CC':
    feats[0, 4] = tags.count('CC')

    # 6
    # if plus_tags[i][1].rstrip(' ') == 'VBD':
    feats[0, 5] = tags.count('VBD')

    # 7
    # if plus_tags[i][0] in ("\'ll", 'will', 'gonna'):
    #     feats[0, 6] += 1
    # if plus_tags[i][0] == 'going' and plus_tags[i + 1][0] == 'to' and plus_tags[i + 2][1] == 'VB':
    #     feats[0, 6] += 1
    feats[0, 6] = len(re.compile(r'\b(' + r'|'.join(["\'ll", 'will', 'gonna']) + r')\b').findall(comment))
    feats[0, 6] += len(re.compile(r"go/VBG to/TO [\w]+/VB").findall(comment))

    # 8
    # if plus_tags[i][1].rstrip(' ') == ',':
    feats[0, 7] = tags.count(',')

    # 10
    # if plus_tags[i][1].rstrip(' ') in ('NN', 'NNS'):
    feats[0, 9] = tags.count('NN')
    feats[0, 9] += tags.count('NNS')

    # 11
    # if plus_tags[i][1].rstrip(' ') in ('NNP', 'NNPS'):
    feats[0, 10] = tags.count('NNP')
    feats[0, 10] += tags.count('NNPS')

    # 12
    # if plus_tags[i][1].rstrip(' ') in ('RB', 'RBR', 'RBS'):
    feats[0, 11] = tags.count('RB')
    feats[0, 11] += tags.count('RBR')
    feats[0, 11] += tags.count('RBS')

    # 13
    # if plus_tags[i][1].rstrip(' ') in ('WDT', 'WP', 'WP$', 'WRB'):
    feats[0, 12] = tags.count('WDT')
    feats[0, 12] += tags.count('WP')
    feats[0, 12] += tags.count('WP$')
    feats[0, 12] += tags.count('WRB')

    for i in range(l):
        # plus_tags[i][0], plus_tags[i][1].rstrip(' ')

        # 1
        if plus_tags[i][0].isupper() and len(plus_tags[i][0]) >= 3:
            feats[0, 0] += 1

        # 9
        is_punctuation = True
        for letter in plus_tags[i][0]:
            if letter not in string.punctuation:
                is_punctuation = False
                break
        if is_punctuation and len(plus_tags[i][0]) >= 2:
            feats[0, 8] += 1

        # 14
        if plus_tags[i][0] in SLANG:
            feats[0, 13] += 1

    # 15, 17
    c_count = comment.count('\n')
    if c_count == 0:
        c_count += 1
    feats[0, 14] = l / c_count
    feats[0, 16] = c_count

    # 16
    letter_sum = 0
    token_sum = 0
    for i in range(l):
        if plus_tags[i][1].rstrip(' ') not in string.punctuation:
            letter_sum += len(plus_tags[i][0])
            token_sum += 1
    if token_sum == 0:
        token_sum += 1
    feats[0, 15] = letter_sum / token_sum

    # 18 - 29
    AoAs = []
    IMGs = []
    FAMs = []
    Vs = []
    Ds = []
    As = []
    inter_words = tuple(set(AoA.keys()).intersection(set(words)))
    inter_words2 = tuple(set(V.keys()).intersection(set(words)))

    for i in range(len(inter_words)):
        if len(inter_words[i]) > 0:  # some /n become /n_SP
            AoAs.append(int(AoA[inter_words[i]]))
            IMGs.append(int(IMG[inter_words[i]]))
            FAMs.append(int(FAM[inter_words[i]]))
            
    for i in range(len(inter_words2)):
        if len(inter_words2[i]) > 0:
            Vs.append(float(V[inter_words2[i]]))
            Ds.append(float(D[inter_words2[i]]))
            As.append(float(A[inter_words2[i]]))

    if len(AoAs) > 0:
        feats[0, 17] = np.mean(AoAs)
        feats[0, 18] = np.mean(IMGs)
        feats[0, 19] = np.mean(FAMs)
        feats[0, 20] = np.std(AoAs)
        feats[0, 21] = np.std(IMGs)
        feats[0, 22] = np.std(FAMs)
    if len(Vs) > 0:
        feats[0, 23] = np.mean(Vs)
        feats[0, 24] = np.mean(Ds)
        feats[0, 25] = np.mean(As)
        feats[0, 26] = np.std(Vs)
        feats[0, 27] = np.std(Ds)
        feats[0, 28] = np.std(As)

    return feats


def extract2(feats, comment_class, comment_id):
    ''' This function adds features 30-173 for a single comment.

    Parameters:
        feats: np.array of length 173
        comment_class: str in {"Alt", "Center", "Left", "Right"}
        comment_id: int indicating the id of a comment

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (this 
        function adds feature 30-173). This should be a modified version of 
        the parameter feats.
    '''
    if comment_class == "Alt":
        id = alt_IDs.index(comment_id)
        feats[29:173] = alt_feats[id]
    elif comment_class == "Center":
        id = center_IDs.index(comment_id)
        feats[29:173] = center_feats[id]
    elif comment_class == "Left":
        id = left_IDs.index(comment_id)
        feats[29:173] = left_feats[id]
    elif comment_class == "Right":
        id = right_IDs.index(comment_id)
        feats[29:173] = right_feats[id]
    return feats


def main(args):
    data = json.load(open(args.input))
    feats = np.zeros((len(data), 173 + 1))

    # TODO: Use extract1 to find the first 29 features for each 
    # data point. Add these to feats.
    for i in range(feats.shape[0]): # feats.shape[0]

        print("Extracting from comment #{}".format(i))
        feats[i, :29] = extract1(data[i]["body"])

        # TODO: Use extract2 to copy LIWC features (features 30-173)
        # into feats. (Note that these rely on each data point's class,
        # which is why we can't add them in extract1).
        if data[i]["cat"] == "Left":
            feats[i][-1] = 0
            feats[i] = extract2(feats[i], data[i]["cat"], data[i]['id'])
        elif data[i]["cat"] == "Center":
            feats[i][-1] = 1
            feats[i] = extract2(feats[i], data[i]["cat"], data[i]['id'])
        elif data[i]["cat"] == "Right":
            feats[i][-1] = 2
            feats[i] = extract2(feats[i], data[i]["cat"], data[i]['id'])
        elif data[i]["cat"] == "Alt":
            feats[i][-1] = 3
            feats[i] = extract2(feats[i], data[i]["cat"], data[i]['id'])

    np.savez_compressed(args.output, feats)
    end_time = datetime.datetime.now()
    print("Start time: {}".format(start_time))
    print("End time: {}".format(end_time))




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    parser.add_argument("-p", "--a1_dir",
                        help="Path to csc401 A1 directory. By default it is set to the cdf directory for the assignment.",
                        default="/u/cs401/A1/")
    args = parser.parse_args()

    main(args)
