#! /usr/bin/env python

import os, glob, shutil, string, pickle, gzip, re

def unpickle(fileName):
    with open(fileName) as f:
        return pickle.load(f)


def get_database_cursor(db_name):
    return 0



def loadData(fileNames):
    ''' (list of files) -> dict of str: list of tuple(int, str)
    '''
    data = {}
    for f in fileNames:
        fh = open(f)
        lines = fh.readlines()
        part = []
        for line in lines:
            line = line.strip().split()
            part.append((int(line[0]), " ".join(line[1:])))
        data[f] = part
        fh.close()
    return data



def collapse(comment):
    out = ""
    prev = None
    prevprev = None
    for c in comment:
        if not c == prev == prevprev:
            out += c
        prev = c
        prevprev = prev
    return out



def tokenize(comment):
    comment = comment.lower()
    comment = collapse(comment)
    return re.findall(r"[\w']+", comment)



def getWordDict(data, stopWords):
    ''' (dict of str: list of tuple(int, str)) ->
            dict of str: dict of int: int

    Takes the output of a call to loadData(fileNames) and build
    the corresponding dictionnary of words.
    Each entry in the dict is a dict with two keys:
        0: nHits in 'good' comments
        1: nHits in 'bad' comments
    '''
    wordDict = {}
    for p in data:
        for sample in data[p]:
            comment = sample[1]
            for word in tokenize(comment):
                if word in wordDict:
                    wordDict[word][sample[0]] += 1
                else:
                    wordDict[word] = {0: 0, 1: 0}
                    wordDict[word][sample[0]] += 1
    return wordDict



def getBigrams(data, stopWords):
    ''' (dict of str: list of tuple(int, str)) ->
            dict of str: dict of int: int

    Takes the output of a call to loadData(fileNames) and build
    the corresponding dictionnary of words.
    Each entry in the dict is a dict with two keys:
        0: nHits in 'good' comments
        1: nHits in 'bad' comments
    '''
    bigramsDict = {}
    for p in data:
        for sample in data[p]:
            comment = sample[1]
            tokens = tokenize(comment)
            for i in range(len(tokens)-1):
                bigram = tokens[i] + tokens[i+1]
                if bigram in bigramsDict:
                    bigramsDict[bigram][sample[0]] += 1
                else:
                    bigramsDict[bigram] = {0: 0, 1: 0}
                    bigramsDict[bigram][sample[0]] += 1
    return bigramsDict



def loadDict(fileName):
    print ("Loading dictionnary from: %s" % fileName)
    stopWords = set()
    with open(fileName) as f:
        for line in f:
            line = line.strip()
            stopWords.add(line)
    return stopWords


def ranges_gen(size = 10, window = 3):
    ''' Generator for train, test set numbers for many-fold evaluation '''
    n = 0
    while True:
        test = [v % size for v in range(n, n + window)]
        train = [v for v in range(size) if v not in test]
        yield (train, test)
        n += 1




def get_ircslug_from_uid(cur, uid):
    ''' (db_cursor, int) -> str

    Returns the name of an user from his uid.

    '''
    q = ("select name_slug_irc,name_slug from users where user_id like '%s'" % uid)
    cur.execute(q)
    res = cur.fetchall()
    if len(res) == 0:
        return None
    return res[0]






def getColorCode(color):
    if color == 'green':
        return 92
    if color == 'red':
        return 91
    if color == 'yellow':
        return 93
    return 99



def colorString(color, string):
    colorCode = getColorCode(color)
    return "\033[{}m{}\033[0m".format(colorCode, string)



def printColor(color, string):
    print (colorString(color, string))


