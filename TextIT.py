from textblob import TextBlob
from nltk.probability import FreqDist
from transformers import pipeline
import matplotlib.pyplot as plt
import click
from nltk.corpus import stopwords
from rich.progress import track
from rich import print
from art import tprint
from rich.console import Console
from os import listdir, mkdir, path


tprint("textit",font="block",chr_ignore=True)
print('Version 1.0\nPowered by Gruzdeva Daria\nFollow on https://github.com/qwertysism/Text-analysis')
console = Console()
@click.command()
@click.option('--file', '-f', help='Выберите файл для анализа')
@click.option('--output', '-o', default='reports', help='Выберите папку для сохранения отчетов')
@click.option('--ptt', '-p', help='Выберите папку с предобработанными текстами')
def init(file, output, ptt):
    if ptt is None:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                text = f.read()
                print(f'\n\nФайл для анализа: {path.basename(file)}')
                with console.status("[bold green]Загрузка модели bart-large-mnli..."):
                    classif = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
                    console.log('[bold][green]Используется модель bart-large-mnli!')
                token_word = tokenize(text)
                labels, scores = classifier(text,classif)
                freq, polarity, subjectivity, unique, all_words = counter(text,token_word)
                print (f'Основной жанр: {labels[0]}\nВторостепенные жанры: {labels[1]}, {labels[2]}')
                report(output,labels, scores,freq, polarity, subjectivity, unique, all_words, path.basename(file))
        except:
            print('[bold][red]Ошибка! Файл не найден или выбран неверный формат файла!')
    else:
        try:
            for count, enty in enumerate(listdir(f'{ptt}/')):
                if path.isfile(f'{ptt}/{enty}'):
                    try:
                        with console.status("[bold green]Загрузка модели bart-large-mnli..."):
                            classif = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
                            console.log('[bold][green]Используется модель bart-large-mnli!')
                        with open(f'{ptt}/{enty}', 'r', encoding='utf-8') as f:
                            text = f.read()
                            print(f'\n\nАнализ {count} файла: {enty}')
                            token_word = tokenize(text)
                            labels, scores = classifier(text,classif)
                            freq, polarity, subjectivity, unique, all_words = counter(text,token_word)
                            print (f'Основной жанр: {labels[0]}\nВторостепенные жанры: {labels[1]}, {labels[2]}')
                            report(output,labels, scores,freq, polarity, subjectivity, unique, all_words,enty)
                    except:
                        print(f'Ошибка при анализе файла {enty}')
        except:
            print('[bold][red]Ошибка! Папка не найдена!')

def classifier(text,classif):
    candidate_labels = ["Adventure", "Crime and Detective", "Drama", "Fairy Tale", "Fantasy", "Historical Fiction", "Horror", "Magical Realism", "Mystery", "Romance", "Satire", "Science Fiction", "Thriller", "Tragedy", "Young adult"]
    with console.status("[bold green]Классификация текста..."):
        classifier=classif(text, candidate_labels)
        console.log('[bold][green]Классификация завершина!')
    labels = classifier['labels']
    scores = classifier['scores']
    return labels, scores

def tokenize(text):
    text = text.replace("\n", " ")
    text = TextBlob(text.lower()) #применяем к тексту библиотеку Textblob для дальнейшей работы и понижаем регистр
    with console.status("[bold green]Токенизация текста..."):
        tokenized = text.words
        console.log('[bold][green]Токенизация завершина!')
    stop_words = set(stopwords.words('english'))
    stop_words.add("would")
    stop_words.add("could")
    stop_words.add("'s")
    stop_words.add("n't")
    stop_words.add("'ll")
    stop_words.add("'m")
    stop_words.add("'ve")
    stop_words.add("'re")
    marks = '''©!()-[]–{}—;?@#$«»%:'"\,./^&amp;*_“”’'''
    exit = []
    for word in tokenized:
        if word not in stop_words and word not in marks:
            exit.append(word)
    return exit


def counter(text,token):
    #вычисляем количество уникальных слов в тексте
    text = text.lower()
    words = text.split()
    words = [word.strip('.,!;()[]') for word in words]
    words = [word.replace("'s", '') for word in words]
    unique = []
    for word in track(words,description='[green]Подсчет уникальных слов'):
        if word not in unique:
            unique.append(word)
    #топ 10 слов по частоте использования
    fdist = FreqDist(token)
    top=fdist.most_common(10)
    freq=[]
    for i in top:
        freq.append(i[0])
    print(f'Всего слов в тексте: {len(words)}\nИз них уникальных: {len(unique)}\nТоп 10 слов на основе частотного анализа: {freq}')
    #Полярность и субъективность текста
    text = TextBlob(str(token))
    polarity = round(text.sentiment.polarity,3)
    if polarity > 0:
        print(f'Положительный со значением {polarity}')
    elif polarity < 0:
        print(f'Отрицательный со значением {polarity}')
    else:
        print(f'Нейтральный со значением {polarity}')
    subjectivity = round(text.sentiment.subjectivity,3)
    print(f'Субъективность со значением {subjectivity}')
    return freq, polarity, subjectivity, len(unique), len(words)




def report(output,labels, scores,freq, polarity, subjectivity, unique, all_words,enty):
    if output in listdir('.'):
    #save to file        
        with open(f'{output}/{enty}_report.md', 'w', encoding='utf-8') as f:
            f.write(f'Всего слов в тексте: {all_words}\nИз них уникальных: {unique}\n\
            Топ 10 слов на основе частотного анализа: {freq}\n\
            Положительный со значением {polarity}\nСубъективность со значением {subjectivity}\n\
            Топ 3 жанра: {labels[0]}, {labels[1]}, {labels[2]}\n\
            Результаты классификации:')
        f.close()
        plt.pie(scores[:3], labels=labels[:3], autopct='%1.1f%%')
        plt.title('Результаты классификации')
        plt.show()
    else:
        mkdir(f'{output}')
        with open(f'{output}/{enty}_report.txt', 'w', encoding='utf-8') as f:
            f.write(f'Всего слов в тексте: {all_words}\nИз них уникальных: {unique}\nТоп 10 слов на основе частотного анализа: {freq}\nПоложительный со значением {polarity}\nСубъективность со значением {subjectivity}\nТоп 3 жанра: {labels[0]}, {labels[1]}, {labels[2]}\nЖанры в убывающем порядке:{labels}\nЗначения жанров: {scores}')
        f.close()
        plt.pie(scores[:3], labels=labels[:3], autopct='%1.1f%%')
        plt.title('Результаты классификации')
        plt.show()






if __name__ == '__main__':
    init()
