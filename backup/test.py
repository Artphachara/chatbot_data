from pythainlp.corpus.common import thai_words
from pythainlp.util import dict_trie, find_keyword, countthai, eng_to_thai
from pythainlp.tokenize import Tokenizer, sent_tokenize, word_tokenize, subword_tokenize
from pythainlp.tokenize.multi_cut import find_all_segment, mmcut, segment
from pythainlp.spell import spell,correct, NorvigSpellChecker
from pythainlp.tag import pos_tag
from pythainlp.tag.named_entity import ThaiNameTagger
import json
import itertools
import pandas as pd
import pythainlp
import random

from textblob import Sentence

with open('insurance_data.json', encoding="utf8") as insurance_data:
  insurance_data = json.load(insurance_data)
with open('lineman_data.json' ,encoding="utf8") as lineman_data:
  lineman_data = json.load(lineman_data)
with open('define_keyword.json' , encoding="utf8") as DefineKeywordData:
  DefineKeywordData = json.load(DefineKeywordData)

#Add Dictionarie
import backup.add_dict as add_dict
import backup.keywording as keywording
import backup.ignoring as ignoring
import backup.run as run
#KEYWORD EXTRACTION
# sentence.
# text = "สนใจรายได้เสริมอาทิตย์ละ 7000-8000 บาทต่อสัปดาห์ติดต่อ @line: phachara1998"
text = input("Test: ")
# eng to thai
text = run.check_eng_to_thai(text)
# get copy sentence of text value
before_text = text
# replace specific word to same word
text = run.defination_word(text)
# tokenize sentence to word in text.
text = word_tokenize(text, custom_dict=add_dict.trie, keep_whitespace=False, engine='newmm')
print(text)
# filter ignore word in text.
classtify_word = []
for ignore_word in text:
# if have ignore word in text, ignore and continue.
    if ignore_word in ignoring.ignore_dict:
        continue
    # if don't have ignore word in text, append word in classtify_word and fill in text.
    else:
        classtify_word.append(ignore_word)
    # collect and display word is have filter by ignore word.
    text = classtify_word
    print(text)

score = []
for element in text:
    # set up point equal 0 for reset point every loop.
    point = 0
    for sub_word in element:
        # tokenize sub_word and count alphabet in sub_word.
        subword_tokenize(sub_word, engine='etcc')
        point = point + 1
    # if element in keywordd, element will get point equal 5.
    if element in keywording.keywordd:
    # print(element)
        point = point + 5
    # get total score of each element.
        score.append(point)
# create dataframe from text and score.
df = pd.DataFrame(list(zip(text, score)),
            columns =['Word', 'Value'])
# group by word and sort by value from maximum to lower and sum score is word have the same name and display only top 3 of value.
df = df.groupby(['Word']).sum().sort_values(by=['Value'], ascending=False).head(3)
print(df)

#MULTI_INTENT
#transform dataframe to list.
df = df.drop(['Value'], axis=1).reset_index().values.tolist()
#unlist in list of keyword.
keyword = list(itertools.chain(*df))
print(keyword)
# looping word in keyword for checking tag in intent.
tag = []
for word in keyword:
    # use insurance_data['intents'] for insurance and use lineman_data for lineman
    for intent in lineman_data['intents']:
        # print(intent['tag'])
        for pattern in intent['patterns']:
            # print(pattern)
            validate = word_tokenize(pattern, custom_dict=add_dict.trie, keep_whitespace=False, engine='newmm')
            # print(validate)
            if word in validate:
                print(word)
                tag.append(intent['tag'])
                print(intent['tag'])
                break

# create score 1 point 3 element in list for merge same word and sort by maximum to lower score.
score = [1, 1, 1]
df = pd.DataFrame(list(zip(tag, score)),
            columns =['Word', 'Value'])
df = df.groupby(['Word']).sum().sort_values(by=['Value'], ascending=False).head(3)
print(df)

# drop value for prepare to use tag.
df = df.drop(['Value'], axis=1).reset_index().values.tolist()
# unlist in list of tag.
keyword_tag = list(itertools.chain(*df))
count = len(keyword_tag)
print(keyword_tag)
# check element taging is equal 3
if count == 3:
# drop greeting tag
    if "greeting" in keyword_tag:
        keyword_tag.remove("greeting")
    print(keyword_tag)
# sorting goodbye tag to the last tag
keyword_tag = run.sort_goodbye(keyword_tag)
# for get first of tag is greeting
first_tag = []
# for get second tag or over before tag greeting
other_tag = []
# check greeting is have in keyword_tag
if "greeting" in keyword_tag:
    # get element from keyword_tag into first element
    for valid in keyword_tag:
        # get greeting tag to the first of tag
        if valid == "greeting":
            first_tag.append(valid)
        # get second or over to other tag for divide from second tag
        else:
            other_tag.append(valid)
        # divide first tag and second tag for the greeting tag is first
        keyword_tag = first_tag + other_tag
print(keyword_tag)
# value sentence before clean
print(before_text)
# checking question with respone (sentence after clean)
print(run.list_to_string(text))
# get keyword_tag in word for find responses in tag with random answer.
for word in keyword_tag:
    for intent in lineman_data['intents']:
        if word == intent['tag']:
        # if intent['tag'] == "greeting":
        #   print("papaya salad")
        # elif intent['tag'] == "goodbye":
        #   print("nongwa")
        # else:
            print(random.choice(intent['responses']))
if word not in keyword_tag:
    print("ขอโทษนะจ๊ะ ไม่พบคำตอบที่คุณตามหา..")

