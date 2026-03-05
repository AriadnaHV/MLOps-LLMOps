# LIBRARIES/MODULES USED IN THIS FILE

# Standard libraries
import json
from typing import Any, Dict
import datetime 

# Data handling
import numpy as np
import pandas as pd

# Natural Language Processing: spaCy
import spacy
from spacy.tokens import Token

# Load spaCy model or download it if needed
try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# Natural Language Processing: nltk
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

#===========================================================================================

# Create stopword set, keeping sentiment/contrast words
base_stopwords = set(stopwords.words("english"))
keep_words = {
    "no", "not", "nor", "but","less","least", "ain'", "above", "below",
    "down", "up", "could", "would", "should", "must", "off", "once", "over", "under"
    "very","too", "like", "less", "least", "more","most","only","just","same","again"
}
stop_words = base_stopwords - keep_words  # Remove from stopwords so they are retained

#===========================================================================================

def preprocess_reviews(df, text_col='text', nlp=nlp, stop_words=stop_words, max_tokens=512):
    """
    Preprocesses text reviews in a DataFrame:
    - Lemmatizes and lowercases
    - Removes punctuation, spaces, and stopwords
    - Truncates long reviews
    - Adds processed text and original token lengths to the DataFrame

    Args:
        df (pd.DataFrame): DataFrame containing the text column
        text_col (str): Name of the column with raw text
        nlp (spacy.lang): spaCy NLP pipeline
        stop_words (set/list): Stopwords to remove
        max_tokens (int, optional): Maximum number of tokens per review. Default is 200.

    Returns:
        pd.DataFrame: Original DataFrame with added columns 'processedText' and 'reviewLength'
    """
    processed_texts = []
    review_lengths = []
    count = 0

    for doc in nlp.pipe(df[text_col], batch_size=1000):
        tokens = [
            token.lemma_.lower()
            for token in doc
            if not token.is_punct and not token.is_space and token.lemma_.lower() not in stop_words
        ]

        review_lengths.append(len(tokens))

        # Truncate if needed
        if len(tokens) > max_tokens:
            tokens = tokens[:(max_tokens // 2)] + tokens[-(max_tokens // 2):]

        processed_texts.append(" ".join(tokens))

        if count % 10000 == 0:
            print(f"{count} reviews preprocessed")
        count += 1

    print("\nPreprocessing finished. Adding preprocessed data to DataFrame.")

    df["reviewLength"] = review_lengths
    df["processedText"] = processed_texts

    print("\nColumns 'reviewLength' and 'processedText' added to DataFrame.")
    return df

#===========================================================================================

def reviews_to_dict(dirpath: Any, filename: Any) -> Dict[int, dict]:
    '''
    - Opens json filepath given and reads each line. 
    - For each line, creates a 'review' dictionary with fields: 
        'label': 1 if negative sentiment (less than 3 stars), 0 otherwise
        'helpful': same list of two integers as in 'helpful'
        'age': review age in days since 07 24, 2014 (all reviews will be at least 1 day old)
        'text': text of the review
    - Each line is itself encoded as an integer entry of the main dictionary 
      (from entry 0 to n-1, where n is the total number of reviews).
    - The final dictionary is saved into a file of the same name with prefix "dict_"

    Arg: 
       path of the json file that contains the reviews
    Returns: 
       dictionary of dictionaries named reviews_dict.
    '''
    reviews_dict = {}
    idx = 0
    dict_file = dirpath + "dict_" + filename
    with open(dirpath+filename, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line)
                
                # Binarize 'overall' score: <3 = 1 (negative), >=3 = 0 (positive)
                if entry.get('overall',3) <= 3:
                    negSentiment = 1 
                else:
                    negSentiment = 0
                
                # Select review text (use 'summary' if longer than 'reviewText')
                reviewText = entry.get('reviewText', '').strip()
                summary = entry.get('summary', '').strip()
                if len(summary) > len(reviewText):
                    reviewText = summary
                
               # Select 'unixReviewTime' and convert to 'reviewAge'
                tmax = humanTime_to_unixTime('07 24, 2014')  # one day after most recent date
                unixReviewTime = entry.get('unixReviewTime')
                if unixReviewTime is None:
                    reviewAge = None
                else:
                    reviewAge = (tmax - unixReviewTime) // (60*60*24)  # convert to days

                # Create dictionary entry for current review and store in main dictionary
                review = {
                    'label': negSentiment,
                    'helpful': entry.get('helpful', [0, 0]),
                    'age': reviewAge,
                    'text': reviewText
                }
                reviews_dict.update({idx: review})
                idx += 1
                
            except json.JSONDecodeError:
                continue
    
    print("Dictionary created.")
    
    with open(dict_file, "w") as f:
        json.dump(reviews_dict, f)
    
    print(f"Dictionary saved to: {dirpath+filename}.")
    print("\nTo load the file, use this code:")
    print(f"\nwith open('{dirpath+filename}', 'r') as f:")
    print(f"  reviews_dict = json.load(f)")
    
    return reviews_dict
    
#===========================================================================================

def unixTime_to_humanTime(unixTime):
    '''
    Converts between unix time (number of seconds since Jan 1, 1970)
    and human time (in the format YYYY MM DD).
    '''
    return datetime.datetime.fromtimestamp(unixTime).strftime('%Y %m %d')
    
#===========================================================================================

def humanTime_to_unixTime(humanTime):
    '''
    Converts between human time (in the format MM DD, YYYY)
    and unix time (number of seconds since Jan 1, 1970).
    '''
    # Parse the date from format mm dd, yyyy
    dt = datetime.datetime.strptime(humanTime, '%m %d, %Y')
    # Convert to Unix time
    return int(dt.timestamp())

#===========================================================================================

