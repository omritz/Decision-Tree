# Decision-Tree

## A decision tree to predict if articles will go viral ##


In this exercise I use a decision tree to predict if articles will go viral, with very
little information on what they actually say. In order to do this, I use the data
file OnlineNewsPopularity.data.
The description of the data file is found in the Online News Popularity names file.

Since many of these variables are continuous, I need to set thresholds to “bucket” them – which means each of the trees
might break different than someone else’s (which is fine!).
Please ignore the first and second attributes which are not relevant for us (the
URL changes for each article, and the number of days since publication is at least a
week for all articles, and since we want it to go viral quickly, we will not know this
in advance). The LDA attributes are measuring how close an article is to the 5 most
popular topics on the website (which have been numbered 0 to 4).
When I build a decision tree, I used entropy to calculate
the more meaningful attributes, and χ2 test to prune vertices.

### There are three importent functions: ###

- buildTree(k) k∈[0, 1] - build a decision tree, using k ratio of
the data (so if k = 0.6,  arbitrarily choose 60% of the data), and validate it
on the remainder. The outcome is printing out the decision tree, and reporting
the error.
  
- treeError(k) k∈N - report the quality of the decision tree by building
k-cross validation, and reporting the error.
  
- isThisViral(<array>) - receive an an input from the user of the article they
are considering. assume it is in the same order as the data file (but
without the bit saying how many shares it got, of course). You return 1 if you
think it will go viral, and 0 if not.

**Try Yourself!**
