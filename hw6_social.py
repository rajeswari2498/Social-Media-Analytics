"""
Social Media Analytics Project
Name:
Roll Number:
"""

import hw6_social_tests as test

project = "Social" # don't edit this

### PART 1 ###

import pandas as pd
import nltk
nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
endChars = [ " ", "\n", "#", ".", ",", "?", "!", ":", ";", ")" ]

'''
makeDataFrame(filename)
#3 [Check6-1]
Parameters: str
Returns: dataframe
'''
def makeDataFrame(filename):
    politicaldata= pd.read_csv(filename)
    return politicaldata


'''
parseName(fromString)
#4 [Check6-1]
Parameters: str
Returns: str
'''
def parseName(fromString):
    first_name=fromString.find(" ")
    slicing1_name=fromString[first_name:]
    last_name=slicing1_name.find("(")
    slicing2_name=slicing1_name[:last_name]
    slicing2_name=slicing2_name.strip()
    # print(slicing2_name)
    return slicing2_name


'''
parsePosition(fromString)
#4 [Check6-1]
Parameters: str
Returns: str
'''
def parsePosition(fromString):
    first_position=fromString.find("(")
    slicing1_position=fromString[first_position+1:]
    last_position=slicing1_position.find(" from")
    slicing2_position=slicing1_position[:last_position]
    slicing2_position=slicing2_position.strip()
    # print(slicing2_position)
    return slicing2_position


'''
parseState(fromString)
#4 [Check6-1]
Parameters: str
Returns: str
'''
def parseState(fromString):
    first_state=fromString.find("from ")
    slicing1_state=fromString[first_state+4:]
    last_state=slicing1_state.find(") ")
    slicing2_state=slicing1_state[:last_state]
    slicing2_state=slicing2_state.strip()
    # print(slicing2_state)
    return slicing2_state


'''
findHashtags(message)
#5 [Check6-1]
Parameters: str
Returns: list of strs
'''
def findHashtags(message):
    list=[]
    find_tag=message.split("#")
    # print(find_tag)
    for hashtag in find_tag[1:]:
        empty_string="#"
        for char in hashtag:
            if char not in endChars:
                empty_string= empty_string+char
            else:
                break
        list.append(empty_string)
    # print(list)
    return list


'''
getRegionFromState(stateDf, state)
#6 [Check6-1]
Parameters: dataframe ; str
Returns: str
'''
def getRegionFromState(stateDf, state):
    row = stateDf.loc[stateDf['state'] == state, 'region']
    # print(row)
    return row.values[0]


'''
addColumns(data, stateDf)
#7 [Check6-1]
Parameters: dataframe ; dataframe
Returns: None
'''
def addColumns(data, stateDf):
    names=[]
    positions=[]
    states=[]
    regions=[]
    hashtags=[]
    for index, row in data.iterrows():
        value=row["label"]
        names.append(parseName(value))
        positions.append(parsePosition(value))
        States_var=parseState(value)
        states.append(States_var)
        regions.append(getRegionFromState(stateDf,States_var))
        value1=row["text"]
        hashtags.append(findHashtags(value1))
    data["name"]=names
    data["position"]=positions
    data["state"]=states
    data["region"]=regions
    data["hashtags"]=hashtags
    return


### PART 2 ###

'''
findSentiment(classifier, message)
#1 [Check6-2]
Parameters: SentimentIntensityAnalyzer ; str
Returns: str
'''
def findSentiment(classifier, message):
    score = classifier.polarity_scores(message)['compound']
    if score < -0.1:
        return "negative"
    if score > 0.1:
        return "positive"
    else:
        return "neutral"


'''
addSentimentColumn(data)
#2 [Check6-2]
Parameters: dataframe
Returns: None
'''
def addSentimentColumn(data):
    classifier = SentimentIntensityAnalyzer()
    sentiments=[]
    for index, row in data.iterrows():
        value_sentiment=row["text"]
        sentiment_pass=findSentiment(classifier,value_sentiment)
        sentiments.append(sentiment_pass)
    data["sentiment"]=sentiments
    # print(data["sentiment"])
    return


'''
getDataCountByState(data, colName, dataToCount)
#3 [Check6-2]
Parameters: dataframe ; str ; str
Returns: dict mapping strs to ints
'''
def getDataCountByState(data, colName, dataToCount):
    count_dictionary={}
    if dataToCount== "" and colName == "":
        for index, row in data.iterrows():
            if row["state"] not in count_dictionary:
                count_dictionary[row["state"]]=1
            else:
                count_dictionary[row["state"]]+=1
    else:
        for index,row in data.iterrows():
            if dataToCount == row[colName]:
                if row["state"] not in count_dictionary:
                    count_dictionary[row["state"]]=1
                else:
                    count_dictionary[row["state"]]+=1
    return count_dictionary


'''
getDataForRegion(data, colName)
#4 [Check6-2]
Parameters: dataframe ; str
Returns: dict mapping strs to (dicts mapping strs to ints)
'''
def getDataForRegion(data, colName):
    nested_dictionary={}
    for index,row in data.iterrows():
        nested_dictionary[row["region"]]={}
    for index,row in data.iterrows():
        if row[colName] not in nested_dictionary[row["region"]]:
            nested_dictionary[row["region"]][row[colName]]=1
        else:
            nested_dictionary[row["region"]][row[colName]]+=1
    return nested_dictionary

'''
getHashtagRates(data)
#5 [Check6-2]
Parameters: dataframe
Returns: dict mapping strs to ints
'''
def getHashtagRates(data):
    hashtag_dictionary={}
    for index,row in data.iterrows():
        for each in row["hashtags"]:
            if each not in hashtag_dictionary:
                hashtag_dictionary[each]=1
            else:
                hashtag_dictionary[each]+=1
    return hashtag_dictionary


'''
mostCommonHashtags(hashtags, count)
#6 [Check6-2]
Parameters: dict mapping strs to ints ; int
Returns: dict mapping strs to ints
'''
def mostCommonHashtags(hashtags, count):
    return


'''
getHashtagSentiment(data, hashtag)
#7 [Check6-2]
Parameters: dataframe ; str
Returns: float
'''
def getHashtagSentiment(data, hashtag):

    return


### PART 3 ###

'''
graphStateCounts(stateCounts, title)
#2 [Hw6]
Parameters: dict mapping strs to ints ; str
Returns: None
'''
def graphStateCounts(stateCounts, title):
    import matplotlib.pyplot as plt
    return


'''
graphTopNStates(stateCounts, stateFeatureCounts, n, title)
#3 [Hw6]
Parameters: dict mapping strs to ints ; dict mapping strs to ints ; int ; str
Returns: None
'''
def graphTopNStates(stateCounts, stateFeatureCounts, n, title):
    return


'''
graphRegionComparison(regionDicts, title)
#4 [Hw6]
Parameters: dict mapping strs to (dicts mapping strs to ints) ; str
Returns: None
'''
def graphRegionComparison(regionDicts, title):
    return


'''
graphHashtagSentimentByFrequency(data)
#4 [Hw6]
Parameters: dataframe
Returns: None
'''
def graphHashtagSentimentByFrequency(data):
    return


#### PART 3 PROVIDED CODE ####
"""
Expects 3 lists - one of x labels, one of data labels, and one of data values - and a title.
You can use it to graph any number of datasets side-by-side to compare and contrast.
"""
# def sideBySideBarPlots(xLabels, labelList, valueLists, title):
#     import matplotlib.pyplot as plt

#     w = 0.8 / len(labelList)  # the width of the bars
#     xPositions = []
#     for dataset in range(len(labelList)):
#         xValues = []
#         for i in range(len(xLabels)):
#             xValues.append(i - 0.4 + w * (dataset + 0.5))
#         xPositions.append(xValues)

#     for index in range(len(valueLists)):
#         plt.bar(xPositions[index], valueLists[index], width=w, label=labelList[index])

#     plt.xticks(ticks=list(range(len(xLabels))), labels=xLabels, rotation="vertical")
#     plt.legend()
#     plt.title(title)

#     plt.show()

# """
# Expects two lists of probabilities and a list of labels (words) all the same length
# and plots the probabilities of x and y, labels each point, and puts a title on top.
# Expects that the y axis will be from -1 to 1. If you want a different y axis, change plt.ylim
# """
# def scatterPlot(xValues, yValues, labels, title):
#     import matplotlib.pyplot as plt
#     fig, ax = plt.subplots()

#     plt.scatter(xValues, yValues)

#     # make labels for the points
#     for i in range(len(labels)):
#         plt.annotate(labels[i], # this is the text
#                     (xValues[i], yValues[i]), # this is the point to label
#                     textcoords="offset points", # how to position the text
#                     xytext=(0, 10), # distance from text to points (x,y)
#                     ha='center') # horizontal alignment can be left, right or center

#     plt.title(title)
#     plt.ylim(-1, 1)

#     # a bit of advanced code to draw a line on y=0
#     ax.plot([0, 1], [0.5, 0.5], color='black', transform=ax.transAxes)

#     plt.show()


### RUN CODE ###

# This code runs the test cases to check your work
if __name__ == "__main__":
    # test.testMakeDataFrame()
    # test.testParseName()
    # test.testParsePosition()
    # test.testParseState()
    # test.testFindHashtags()
    # test.testGetRegionFromState()
    # test.testAddColumns()
    print("\n" + "#"*15 + " WEEK 1 TESTS " +  "#" * 16 + "\n")
    test.week1Tests()
    print("\n" + "#"*15 + " WEEK 1 OUTPUT " + "#" * 15 + "\n")
    test.runWeek1()
    # df = makeDataFrame("data/politicaldata.csv")
    # stateDf = makeDataFrame("data/statemappings.csv")
    # addColumns(df, stateDf)

    test.testFindSentiment()
    test.testAddSentimentColumn()
    df = makeDataFrame("data/politicaldata.csv")
    stateDf = makeDataFrame("data/statemappings.csv")
    addColumns(df, stateDf)
    addSentimentColumn(df)
    test.testGetDataCountByState(df)
    test.testGetDataForRegion(df)
    test.testGetHashtagRates(df)
    ## Uncomment these for Week 2 ##
    """print("\n" + "#"*15 + " WEEK 2 TESTS " +  "#" * 16 + "\n")
    test.week2Tests()
    print("\n" + "#"*15 + " WEEK 2 OUTPUT " + "#" * 15 + "\n")
    test.runWeek2()"""

    # ## Uncomment these for Week 3 ##
    # """print("\n" + "#"*15 + " WEEK 3 OUTPUT " + "#" * 15 + "\n")
    # test.runWeek3()"""
