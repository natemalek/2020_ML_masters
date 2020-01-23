
Arabic Hashtag Lexicon
Version 0.1
30 September 2015
Copyright (C) 2015 National Research Council Canada (NRC)
Contact: Mohammad Salameh (msalameh@ualberta.ca)
	 Saif M. Mohammad (saif.mohammad@nrc-cnrc.gc.ca)		 
	 Svetlana Kiritchenko (Svetlana.Kiritchenko@nrc-cnrc.gc.ca)

Terms of use:
1. This dataset can be used freely for research purposes. 
2. The papers listed below provide details of the creation and use of 
   the dataset. If you use a dataset, then please cite the associated 
   papers.
3. If you use the dataset in a product or application, then please 
   credit the authors and NRC appropriately. Also, if you send us an 
   email, we will be thrilled to know about how you have used the 
   dataset.
4. National Research Council Canada (NRC) disclaims any responsibility 
   for the use of the dataset and does not provide technical support. 
   However, the contact listed above will be happy to respond to 
   queries and clarifications.
5. Rather than redistributing the data, please direct interested 
   parties to this page:
   http://www.purl.org/net/ArabicSA

Please feel free to send us an email:
- with feedback regarding the datasets. 
- with information on how you have used the dataset. 
- if interested in a collaborative research project.

.......................................................................

Arabic Hashtag Lexicon
----------------------

This lexicons is a list of Arabic terms and their associations with
sentiment. The association is shown in the form of number (sentiment
score).  If the score for x1 is greater than the score for x2, then x1
is considered more positive than x2. The lexicon has 13,118 positive
terms and 8,846 negative terms.

The lexicon is extracted automatically from tweets that have certain
seed terms. The seeds used to create this lexicon are a set of 230
Arabic words that were manually selected for being highly positive or
highly negative. For the purposes of creating this lexicon, a tweet is
considered positive if it has a positive seed, and negative if it has
a negative seed.

Sentiment score is calculated using the pointwise mutual information
(PMI) between the term and the positive and negative categories:
      SenScore (w) = PMI(w, pos) − PMI(w, neg) 
where w is a term in the lexicon. PMI(w, pos) is the PMI score between
w and the positive class, and PMI(w, neg) is the PMI score between w
and the negative class. Scores greater than 0 indicate that the term
has a higher tendency to co-occur with positive seeds than negative
seeds---and thus likely positive. Scores less than 0 indicate that the
term has a higher tendency to co-occur with negative seeds than
positive seeds---and thus likely negative.

.......................................................................

PUBLICATIONS
------------
Details of the lexicon and its use in an Arabic sentiment analysis 
system can be found in the following peer-reviewed publications:

- Sentiment After Translation: A Case-Study on Arabic Social Media
Posts.  Mohammad Salameh, Saif M. Mohammad and Svetlana Kiritchenko,
In Proceedings of the North American Chapter of the Association for
Computational Linguistics (NAACL-2015), June 2015, Denver, Colorado.
 
- How Translation Alters Sentiment. Saif M. Mohammad, Mohammad
Salameh, and Svetlana Kiritchenko, In Journal of Artificial
Intelligence Research, in press.

The papers are available here:
http://saifmohammad.com/WebPages/WebDocs/arabicSA-JAIR.pdf
http://aclweb.org/anthology/N/N15/N15-1078.pdf

.......................................................................

VERSION INFORMATION
-------------------
Version 0.1 is the first version as of 30 September 2015.

.......................................................................

FORMAT
------

The lexicon has 5 columns separated by TAB:
— Arabic term: the Arabic term whose sentiment score is provided
— Buckwalter: the Buckwalter transliteration of Arabic terms with Alif
and Ya normalization (different forms of Alif and Ya converted to bare
Aif and dotless Ya).
— Sentiment Score: a score indicating the degree of association of the
Arabic term with positive sentiment.
— Positive Occurrence Count: the number of times the Arabic lemma
co-occurs with positive seeds in the tweets corpus.
— Negative Occurrence Count: the number of times the Arabic lemma
co-occurs with negative seeds in the tweets corpus.
                            
.......................................................................
