import requests
import spacy
import csv
import json
from tqdm import tqdm
from nltk import tokenize
from nltk.stem import PorterStemmer
import re
from nltk.corpus import stopwords



with open("sentences.csv", "r", encoding="utf-8") as f:
    data = csv.DictReader(f=f, delimiter=",")
    data = list(data)

with open("visual_works.csv", encoding="utf-8") as f2:
    visual_works = list(csv.reader(f2, delimiter=","))

with open("buildings.csv", encoding="utf-8") as f3:
    buildings = list(csv.reader(f3, delimiter=","))


nlp = spacy.load("en_core_web_trf")

ps = PorterStemmer()
stopwords_en = set(stopwords.words('english'))
fields= ["sent_id", "start_pos", "end_pos", "surface", "aat_id", "aat_label", "aat_type", "vn_id"]

URL = "http://localhost:8080/predict/semantics"

output = []
pbar=tqdm(total=len(data))
for input in data:
    if input["id"]=="16":
        sentence = input["sentence"]    
        doc = nlp(sentence)
        PARAMS = {'utterance':sentence}
        r = requests.get(url = URL, params = PARAMS)
        response = r.json()
        sentence_lst = tokenize.sent_tokenize(sentence)
        for num, sm_sentence in enumerate(sentence_lst):
            doc = nlp(sm_sentence)
            phrases = set() 
            for nc in doc.noun_chunks:
                if num==0:
                    phrases.add((nc.text, nc.start, nc.end))
                    phrases.add((doc[nc.root.left_edge.i:nc.root.right_edge.i+1].text,\
                    doc[nc.root.left_edge.i:nc.root.right_edge.i+1].start, \
                    doc[nc.root.left_edge.i:nc.root.right_edge.i+1].end))
                else:
                    phrases.add((nc.text, nc.start+len(sentence_lst[num-1]), nc.end+len(sentence_lst[num-1])))
                    phrases.add((doc[nc.root.left_edge.i:nc.root.right_edge.i+1].text,\
                    doc[nc.root.left_edge.i:nc.root.right_edge.i+1].start+len(sentence_lst[num-1]), \
                    doc[nc.root.left_edge.i:nc.root.right_edge.i+1].end+len(sentence_lst[num-1])))
            PARAMS = {'utterance':sm_sentence}
            try:
                r = requests.get(url = URL, params = PARAMS)
                response = r.json()
                for prop in response["props"]:
                    for span in prop["spans"]:
                        if span["vn"] in {"Product", "Result", "Patient"}:
                            if num==0:
                                start_pos = doc[int(span["start"]):int(span["start"]+1)].start
                                end_pos = doc[int(span["end"]):int(span["end"]+1)].end
                            else: 
                                start_pos = doc[int(span["start"]):int(span["start"]+1)].start+len(sentence_lst[num-1])
                                end_pos = doc[int(span["end"]):int(span["end"]+1)].end+len(sentence_lst[num-1])
                            vn = span["vn"]
                            for noun_phrase in phrases:
                                if len(set(range(noun_phrase[1],noun_phrase[2])).intersection(range(start_pos,end_pos)))>0:
                                    tokens = re.findall("\w+", noun_phrase[0])
                                    # removing stopwords
                                    tokens = [word for word in tokens if word not in stopwords_en]
                                    string_1 = " ".join([ps.stem(word) for word in tokens])
                                    for row in visual_works:
                                        aat_id = row[0]
                                        label = row[1]
                                        if label.startswith("<"):
                                            pass
                                        else:
                                            label_no_rounds = re.sub("\s\(.*?\)$", "", label)
                                            string_2 = " ".join([ps.stem(w) for w in label_no_rounds.split(" ")])
                                            if string_1 == string_2:
                                                output.append([input["id"], noun_phrase[1], noun_phrase[2], noun_phrase[0], aat_id, label, "visual_work", vn])
                                    for row in buildings:
                                        aat_id = row[0]
                                        label = row[1]
                                        if label.startswith("<"):
                                            pass
                                        else:
                                            label_no_rounds = re.sub("\s\(.*?\)$", "", label)
                                            string_2 = " ".join([ps.stem(w) for w in label_no_rounds.split(" ")])
                                            if string_1 == string_2:
                                                output.append([input["id"], noun_phrase[1], noun_phrase[2], noun_phrase[0], aat_id, label, "building", vn])
            except KeyError:
                print("error for sentence num", input["id"], num)
        pbar.update(1)
    else:
        pbar.update(1)
        continue



with open('matched_terms', 'w', encoding="utf-8") as f:
    write = csv.writer(f)
    write.writerow(fields)
    write.writerows(output)