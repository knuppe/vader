# VADER - Lexicon file

The lexicon file is a zip file containing 3 files (`language.txt`, `lexicon.tsv` and `emojis.tsv`). This file contains all the rules and definitions that are used in sentiment analysis.


## `language.txt`

This file stores the parameters and rules used in the sentiment evaluator.

Format:

```
rule [tab] mode [tab] operator [tab] valence [tab] stop [tab] sequence [[tab] exception...]
```

### Examples
```r
# comment
rule	<	-	0.293	0	almost * ?

# regex expression for words the s suffix
rule	<	*	-0.74	0	/^.*s$/ * ?
```

| Mode  | Description                                                                  |
| :---: | ---------------------------------------------------------------------------- |
|  `<`  | the rule is evaluated backwards.                                             |
|  `>`  | the rule is evaluated forwards.                                              |
|  `@`  | when the sequence is found in the text the valence is applied in every word. |


| Operator | Description                                      |
| :------: | ------------------------------------------------ |
|   `+`    | Adds value to current valence.                   |
|   `-`    | Subtracts value to current valence.              |
|   `*`    | Multiplies  value to current valence.            |
|   `=`    | Replaces the value with the valence of the rule. |

## `lexicon.tsv`

The lexicon file follows the definition in the [original](https://github.com/cjhutto/vaderSentiment) vader implementation. The only difference is that the emoticons have been moved to the `emojis.tsv` file.

  FORMAT: the file is tab delimited with TOKEN, MEAN-SENTIMENT-RATING, STANDARD DEVIATION, and RAW-HUMAN-SENTIMENT-RATINGS

	NOTE: The current algorithm makes immediate use of the first two elements (token and mean valence). The final two elements (SD and raw ratings) are provided for rigor.  For example, if you want to follow the same rigorous process that we used for the study, you should find 10 independent humans to evaluate/rate each new token you want to add to the lexicon, make sure the standard deviation doesn't exceed 2.5, and take the average rating for the valence. This will keep the file consistent.
	
    DESCRIPTION: 
    Empirically validated by multiple independent human judges, VADER incorporates a "gold-standard" sentiment lexicon that is especially attuned to microblog-like contexts.
    
    The VADER sentiment lexicon is sensitive both the **polarity** and the **intensity** of sentiments expressed in social media contexts, and is also generally applicable to sentiment analysis in other domains.
	
	Sentiment ratings from 10 independent human raters (all pre-screened, trained, and quality checked for optimal inter-rater reliability). Over 9,000 token features were rated on a scale from "[–4] Extremely Negative" to "[4] Extremely Positive", with allowance for "[0] Neutral (or Neither, N/A)".  We kept every lexical feature that had a non-zero mean rating, and whose standard deviation was less than 2.5 as determined by the aggregate of those ten independent raters.  This left us with just over 7,500 lexical features with validated valence scores that indicated both the sentiment polarity (positive/negative), and the sentiment intensity on a scale from –4 to +4. For example, the word "okay" has a positive valence of 0.9, "good" is 1.9, and "great" is 3.1, whereas "horrible" is –2.5, the frowning emoticon :( is –2.2, and "sucks" and it's slang derivative "sux" are both –1.5.
	
    Manually creating (much less, validating) a comprehensive sentiment lexicon is a labor intensive and sometimes error prone process, so it is no wonder that many opinion mining researchers and practitioners rely so heavily on existing lexicons as primary resources. We are pleased to offer ours as a new resource. We began by constructing a list inspired by examining existing well-established sentiment word-banks (LIWC, ANEW, and GI). To this, we next incorporate numerous lexical features common to sentiment expression in microblogs, including:
	
    * a full list of Western-style emoticons, for example, :-) denotes a smiley face and generally indicates positive sentiment
    * sentiment-related acronyms and initialisms (e.g., LOL and WTF are both examples of sentiment-laden initialisms)
    * commonly used slang with sentiment value (e.g., nah, meh and giggly). 
	
    We empirically confirmed the general applicability of each feature candidate to sentiment expressions using a wisdom-of-the-crowd (WotC) approach (Surowiecki, 2004) to acquire a valid point estimate for the sentiment valence (polarity & intensity) of each context-free candidate feature. 

## `emojis.tsv`

The emojis file is tab delimited with emoji and the sentiment rating.

The purpose of emoticons being in a separate file is to improve sentiment analysis performance and also improve portability to new languages.