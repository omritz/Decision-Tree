import pandas as pd
import numpy as np
import math
from scipy.stats import chi2

#    /========\    |\          /| |====\  ===|===
#   /          \   | \        / | |     \    |
#  /            \  |  \      /  | |     /    |
#  \            /  |   \    /   | |====/     |
#   \          /   |    \  /    | |    \     |
#    \========/    |     \/     | |     \ ===|===

""" This script build a decision tree from scratch by recursive functions and data frames from pandas,
 it is take a few seconds to build tree so please be patient """


class TreeNode(object):
    def __init__(self, attribute='leaf', threshold=0, entropy=1, examples=None, childrenLower=[], childrenHigher=[],
                 label=None):
        self.examples = examples  # data in this node
        self.entropy = entropy  # entropy
        self.attribute = attribute  # which attribute is chosen, it non-leaf
        self.children_lower = childrenLower  # child node that lower then threshold
        self.children_higher = childrenHigher  # child node that higher then threshold
        self.threshold = threshold  # the threshold that set for this attribute
        self.label = label  # if it is a leaf, so what will be the answer (viral/ not viral)

    def print(self):
        print('attribute is: ', self.attribute)
        print('threshold is: ', self.threshold)


def makeData():
    """ return data frame with 1 or 0 in the target attribute """
    data_filename = "OnlineNewsPopularity.data.csv"
    df = pd.read_csv(data_filename)
    viral = df.shares >= 2000
    notViral = df.shares < 2000
    df.loc[viral, 'shares'] = 1
    df.loc[notViral, 'shares'] = 0
    df = df.drop('timedelta', axis='columns').drop('url', axis='columns')
    return df


def divideData(dataRatio, set):
    """ split the data set to train set and test set"""
    msk = np.random.rand(len(set)) < dataRatio
    train, test = set[msk], set[~msk]
    return train, test


def getThreshold(attribute, examples):
    """ calculate the threshold by the mean of each attribute"""
    return examples[attribute].mean()


def calcEntropy(attribute, examples, threshold):
    """ calculate the entropy for given attribute"""
    n = len(examples)
    query = examples[attribute] >= threshold
    higher = examples[query]
    lower = examples[~query]
    return len(higher) / n * entropy(higher) + len(lower) / n * entropy(lower)


def entropy(targetSet):
    """ Help func to calculate entropy"""
    n = len(targetSet)
    if n == 0:
        return 0
    viral = len(targetSet[targetSet['shares'] == 1])
    notViral = len(targetSet[targetSet['shares'] == 0])
    if viral == 0 or notViral == 0:
        return 0
    return -(viral / n) * math.log2(viral / n) - (notViral / n) * math.log2(notViral / n)


def pluralityValue(examples):
    """ Checks plurality value - most common value of target in examples"""
    viral = len(examples[examples['shares'] == 1])
    notViral = len(examples[examples['shares'] == 0])
    if viral > notViral:
        return 1
    else:
        return 0


def getRoot(examples):
    """ Return tree node with the attribute that have the min entropy"""
    minEntropy = 1
    attribute = ''
    features = list(examples.columns[0:59])
    for feature in features:
        if feature == 'shares':
            break
        threshold = getThreshold(feature, examples)
        entropy = calcEntropy(feature, examples, threshold)
        if minEntropy > entropy:
            minEntropy = entropy
            attribute = feature
    print('min entropy is ' + attribute + ': ', minEntropy)
    threshold = getThreshold(attribute, examples)
    examplesLower = examples[examples[attribute] < threshold].drop(attribute, axis=1)
    examplesHigher = examples[examples[attribute] >= threshold].drop(attribute, axis=1)
    examples = examples.drop(attribute, axis=1)
    return TreeNode(attribute, threshold, minEntropy, examples, childrenLower=examplesLower,
                    childrenHigher=examplesHigher)


def viralNotViral(examples):
    """ Return the number of viral and not viral examples """
    viral = len(examples[examples['shares'] == 1])
    notViral = len(examples[examples['shares'] == 0])
    return viral, notViral


def pruneVertices(tree):
    """ Prune the tree vertices by Chi^2 test"""
    Kstatisti = 0
    if tree.children_higher.attribute == 'leaf' and tree.children_lower.attribute == 'leaf':  # if is it leaf
        higherExamples = tree.children_higher.examples
        lowerExamples = tree.children_lower.examples
        vH, nvH = viralNotViral(higherExamples)  # num of Viral that higher , num of NotViral that higher
        vL, nvL = viralNotViral(lowerExamples)  # num of Viral that lower , num of NotViral that lower
        probH = (vH + nvH)/(len(higherExamples)+len(lowerExamples))  # probability higher
        probL = (vL + nvL)/(len(higherExamples)+len(lowerExamples))  # probability Lower
        vHN = probH * (vH + vL)
        vLN = probL * (vH + vL)
        nvHN = probH * (nvH + nvL)
        nvLN = probL * (nvH + nvL)
        if vHN != 0:
            Kstatisti = Kstatisti + ((vHN - vH)**2)/vHN
        if nvHN != 0:
            Kstatisti = Kstatisti + ((nvHN - nvH)**2)/nvHN
        if vLN != 0:
            Kstatisti = Kstatisti + ((vLN - vL)**2)/vLN
        if nvLN != 0:
            Kstatisti = Kstatisti + ((nvLN - nvL)**2)/nvLN
        Kcriti = chi2.ppf(0.95, len(higherExamples) + len(lowerExamples) - 1)
        if Kstatisti < Kcriti:
            if vH + vL > nvH + nvL:
                return TreeNode(label=1)
            else:
                return TreeNode(label=0)
        else:
            return tree
    # recursive, until we reach a leaf
    elif tree.children_higher.attribute == 'leaf' and tree.children_lower.attribute != 'leaf':
        tree.children_lower = pruneVertices(tree.children_lower)
    elif tree.children_higher.attribute != 'leaf' and tree.children_lower.attribute == 'leaf':
        tree.children_higher = pruneVertices(tree.children_higher)
    else:
        tree.children_higher = pruneVertices(tree.children_higher)
        tree.children_lower = pruneVertices(tree.children_lower)
    return tree


def decisionTree(examples, parnet_examples):
    """ Recursive func that building the tree, in addition prune nodes that have less then 300 examples"""
    if examples.empty:
        return TreeNode(examples=parnet_examples, label=pluralityValue(parnet_examples))
    if len(examples) < 300:
        return TreeNode(examples=examples, label=pluralityValue(examples))
    elif len(examples) == len(examples[examples['shares'] == 0]):
        return TreeNode(examples=examples, label=0)
    elif len(examples) == len(examples[examples['shares'] == 1]):
        return TreeNode(examples=examples, label=1)
    else:
        root = getRoot(examples)
        examplesHigher = root.children_higher
        examplesLower = root.children_lower
        root.children_higher = decisionTree(examplesHigher, root.examples)
        root.children_lower = decisionTree(examplesLower, root.examples)
        return root


def isThisViral(example, tree):
    """ Recursive func that check if example is viral or not by given tree"""
    if tree.attribute == 'leaf':
        return tree.label
    else:
        if example[tree.attribute] < tree.threshold:
            return isThisViral(example, tree.children_lower)
        else:
            return isThisViral(example, tree.children_higher)


def calcError(tree, testSet):
    """ Calculate the error of given tree and test set """
    answer = 0
    for example in testSet.iterrows():
        if isThisViral(example[1], tree) == example[1]['shares']:
            answer += 1
    return 1 - answer / len(testSet)


def buildTree(ratio):
    data = makeData()
    trainSet, testSet = divideData(ratio, data)
    tree = decisionTree(trainSet, trainSet)
    tree = pruneVertices(tree)
    printTree(tree)
    print('Error: ', calcError(tree, testSet)*100, '%')
    return tree


def printTree(tree, dad=None, underOver=None):
    """ printing given tree , recursive func """
    if tree.attribute == 'leaf':
        print('Dad: ', dad, ', Answer: ', tree.label, ', Under or Over thershold of the dad: ', underOver)
        return
    print('Dad: ', dad, ', Attribute: ', tree.attribute, ', Threshold: ', tree.threshold,
          ', Under or Over thershold of the dad: ', underOver)
    printTree(tree.children_lower, tree.attribute, 'Under')
    printTree(tree.children_higher, tree.attribute, 'Over')


def treeError(k):
    """ calculate tree error by k fold cross validation"""
    data = makeData()
    trainSet, testSet = divideData(1/k, data)
    k_cross_validation(k, trainSet)


def k_cross_validation(k, trainSet):
    dataGroups = np.array_split(trainSet, k)
    totalError = 0
    for i in range(k):
        testSet = dataGroups[i]
        helpList = []
        for j in range(k):
            if j != i:
                helpList.append(dataGroups[j])
        trainSet = pd.concat(helpList)
        tree = decisionTree(trainSet, trainSet)
        totalError += calcError(tree, testSet)
    print("Error for k-cross validation: ", (totalError / k)*100, '%')



if __name__ == "__main__":
    print('start')
    buildTree(0.6)
    treeError(5)

