from textblob import TextBlob
from nltk.probability import FreqDist
from transformers import pipeline
import matplotlib.pyplot as plt

#Чтение файла
file = open('C:/Users/fashi/OneDrive/Рабочий стол/Диплом/ФАЙЛ/Crime and Punishment.txt', 'r', encoding='utf-8') #open text file in read mode
text = file.read() #read whole file to a string

#Определение жанра

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
sequence_to_classify = text
candidate_labels = ["Adventure", "Classic", "Crime and Detective", "Drama", "Fable", "Fairy Tale", "Fan-Fiction", "Fantasy", "Historical Fiction", "Horror", "Humor", "Legend", "Magical Realism", "Mystery", "Mythology", "Romance", "Satire", "Science Fiction", "Short Story", "Thriller", "Tragedy", "Young adult"]
classifier=classifier(sequence_to_classify, candidate_labels)

labels = classifier['labels']
scores = classifier['scores']
a=0
for label in labels:
    score = scores[a]
    print(label+" " + str(round((score*100),2)) +"%")#Выводим жанры и проценты
    a+=1
    if a>2:
        break

#Понижение регистра
text = text.lower()

#Подсчет длины произведения (число слов)
length = len(text.split())
print("Количество слов в тексте: " + str(length))

#Токенизация текста
text = TextBlob(text) #применяем к тексту библиотеку Textblob для дальнейшей работы
tokenized = text.words

#Удаление стоп-слов, знаков препинания и символов
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
stop_words.add("would")
stop_words.add("could")
stop_words.add("'s")
stop_words.add("n't")
stop_words.add("'ll")
marks = '''©!()-[]–{}—;?@#$«»%:'"\,./^&amp;*_“”’'''

exit = []
for word in tokenized:
    if word not in stop_words and word not in marks:
        exit.append(word)

#Частотность
fdist = FreqDist(exit)
rrr=fdist.most_common(10)
new_list=[]
for i in rrr:
    new_list.append(i[0])
print("Топ 10 слов: ", new_list)

#Возвращаем текст в строку (лист-строка)
a=str(exit)

#Полярность и субъективность текста
exit = TextBlob(a)
print("Общая тональность текста (-1:1): ", round(exit.sentiment.polarity,3))  #полярность (-1:1)
print("Общая субъективность текста (0:1): ", round(exit.sentiment.subjectivity,3))  #субъективность (0:1)

#Графики

plt.bar(labels[0:5],scores[0:5])
plt.show()

