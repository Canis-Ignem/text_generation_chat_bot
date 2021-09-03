from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import csv
import re
import os
import codecs
from io import open



# Splits each line of the file into a dictionary of fields
def loadLines(fileName, fields):
    lines = {}

    with open(fileName, 'r', encoding='iso-8859-1') as f:

        for line in f:
            values = line.split(" +++$+++ ")
            lineObj = {}

            for i, field in enumerate(fields):
                lineObj[field] = values[i]

            lines[lineObj['lineID']] = lineObj

    return lines


def loadConversations(fileName, lines, fields):
    conversations = []

    with open(fileName, 'r', encoding='iso-8859-1') as f:

        for line in f:
            values = line.split(" +++$+++ ")
            convObj = {}
        
            for i, field in enumerate(fields):
                convObj[field] = values[i]

            utterance_id_pattern = re.compile('L[0-9]+')
            lineIds = utterance_id_pattern.findall(convObj["utteranceIDs"])
            convObj["lines"] = []

            for lineId in lineIds:
                convObj["lines"].append(lines[lineId])

            conversations.append(convObj)

    return conversations



def extractSentencePairs(conversations):
    qa_pairs = []

    for conversation in conversations:
        for i in range(len(conversation["lines"]) - 1): 
            
            inputLine = conversation["lines"][i]["text"].strip()
            targetLine = conversation["lines"][i+1]["text"].strip()

            if inputLine and targetLine:
                qa_pairs.append([inputLine, targetLine])

    return qa_pairs


def preprocess(p_folder, p_lines, p_conversations, save_pth):

    corpus = p_folder
    datafile = save_pth
    delimiter = '\t'
    delimiter = str(codecs.decode(delimiter, "unicode_escape"))
    lines = {}
    conversations = []
    MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
    MOVIE_CONVERSATIONS_FIELDS = ["character1ID", "character2ID", "movieID", "utteranceIDs"]

    print("\nProcessing corpus...")
    lines = loadLines(os.path.join(corpus, p_lines), MOVIE_LINES_FIELDS)

    print("\nLoading conversations...")
    conversations = loadConversations(os.path.join(corpus, p_conversations),
                                    lines, MOVIE_CONVERSATIONS_FIELDS)

    print("\nWriting newly formatted file...")
    with open(datafile, 'w', encoding='utf-8') as outputfile:
        writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
        for pair in extractSentencePairs(conversations):
            writer.writerow(pair)

