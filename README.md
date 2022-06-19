# TextIT
*Automatic analysis system for fiction in English*

[Google Colab](https://colab.research.google.com/drive/1-J7A-MlO2w-5lWGzOhFlpznkS2_f1vJs?usp=sharing)
## Introduction
In recent decades, many automatic text processing systems have appeared. Mostly such systems are created to work with posts from social networks, comments and reviews. While there are practically no, at least in the public domain, models adapted to work with literary texts.

This program was created as a useful tool for researchers working with texts. Its main goal is to facilitate the work and interaction of researchers with texts.

TextIT performs text processing in English and compiles a report with the data received about it, such as:
- the total volume of the text;
- the number of unique words;
- list of the most frequent words;
- the degree of subjectivity of the text;
- text sentiment;
- the genre of the text.

## Preprocessing
After the file is loaded, the text is normalized, which occurs in the `tokenize()` function. This removes blank lines and lowers the case of the text. The text is split into tokens using the TextBlob library. 

To work with stop words, the program uses the NLTK library. A list of stop words is loaded from this library.
```python
{'as', "shouldn't", 'until', 'can', 'was', 'to', 'no', 'after', 'yourselves', 'shouldn', 'haven', 'myself', 'against', 's', 'them', 'nor', "should've", 'into', "it's", 'below', 'each', 'ma', "couldn't", 'through', 'did', 'should', 'off', 'themselves', 'by', 'my', "doesn't", 'his', 'but', "hadn't", 'which', 'who', 'down', 'won', 'are', 'didn', 'you', 'o', 'himself', 'its', 'such', 'it', 'hasn', 'once', 're', 'there', 'at', 'been', 'aren', 'an', 'm', 'do', "won't", "wasn't", 'under', 'further', "you're", 'then', 'here', 'being', 'than', 'hadn', 'their', 'this', 'your', 'ain', "isn't", 't', 'wouldn', "she's", 'any', 'needn', 'the', 'yours', 'couldn', 'were', 'she', 'him', 'yourself', 'these', 'have', "don't", "mustn't", 'some', 'during', 'own', 'for', 'is', 'again', 'over', "you've", "weren't", 'mightn', 'ours', 'other', 'between', 'very', 'me', 'y', "shan't", 'while', 'above', 'wasn', 'i', 'all', 'herself', "you'll", 'most', "aren't", 'where', 'having', 'shan', "needn't", 'only', 'so', 'before', 'with', 'll', 'we', 'theirs', 'am', "wouldn't", 'hers', 'our', "mightn't", 'and', 'same', 'isn', 'they', 've', 'now', 'doing', 'those', 'few', 'about', "didn't", 'if', 'that', 'how', 'had', 'itself', 'doesn', 'more', 'from', 'in', 'out', "hasn't", 'too', 'just', 'her', 'not', 'whom', 'he', 'will', 'weren', 'd', 'ourselves', 'be', 'when', "that'll", 'a', 'has', 'don', 'mustn', "you'd", 'because', 'of', 'both', 'or', 'up', "haven't", 'on', 'does', 'what', 'why'}
```
## Statistical analysis
Word frequency is calculated by a class such as FreqDist() from the NLTK library. After that, a list of 10 most frequently occurring words is formed.

```python
fdist = FreqDist(token)
top=fdist.most_common(10)
freq=[]
for i in top:
    freq.append(i[0])
```
## Sentiment analisys
Sentiment and subjectivity analysis is implemented using a method based on working with a sentiment dictionary from TextBlob library, where each word has 3 meanings: sentiment, subjectivity, and intensity. For the overall evaluation of the text, the average value of all the words of the text is used.

```python
text = TextBlob(text.lower())
tokenized = text.words
```

## Genre classification
Zero-shot classification is used to determine the genre which allows the model to recognize objects even from those classes that it did not see during training. This work uses the [bart-large-mnli](https://huggingface.co/facebook/bart-large-mnli) model developed by Meta.

```python
classif = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
```

To work, the program needs text and labels, the probability of belonging to which will be calculated. Labels can be not only words, but also phrases and even sentences.
In this work, the following list of genres was chosen as labels:
| Genre |Genre|
| -----|------|
| Adventure | Mystery |
| Crime and Detective|Romance|
| Drama | Satire |
| Fairy Tale | Science Fiction|
| Fantasy|Thriller|
|Historical|Tragedy|
|Horror|Young adult|
|Magical Realism| |

During the report, a chart is generated that reflects the result of the classification.
<p align="left">
  <img width="400" src="https://user-images.githubusercontent.com/98316503/174467897-25124986-2d50-4abc-a75e-54d690e5c7fe.png">
</p>

## Results
Below is a fragment of the pivot table. The works are arranged in ascending order of their sentiment. Works are marked in red, the genres of which are correct or very close to correct.

In general, genre prediction accuracy is about 53%.

![image](https://user-images.githubusercontent.com/98316503/174469805-6f81040a-3950-45ab-a151-8fddaab4dcc0.png)

