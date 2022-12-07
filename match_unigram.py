import requests
import spacy
import csv
import json
from tqdm import tqdm
from nltk import tokenize
from nltk.stem import PorterStemmer
import re

with open("sentences.csv", "r", encoding="utf-8") as f:
    data = csv.DictReader(f=f, delimiter=",")
    data = list(data)

with open("visual_works.csv", encoding="utf-8") as f2:
    visual_works = list(csv.reader(f2, delimiter=","))

with open("buildings.csv", encoding="utf-8") as f3:
    buildings = list(csv.reader(f3, delimiter=","))

with open("components.csv", encoding="utf-8") as f4:
    components = list(csv.reader(f4, delimiter=","))

nlp = spacy.load("en_core_web_trf")

ps = PorterStemmer()
fields= ["sent_id", "start_pos", "end_pos", "surface", "aat_id", "aat_label", "aat_type", "vn_id"]

URL = "http://localhost:8080/predict/semantics"

output = []
pbar=tqdm(total=len(data))
for input in data:
    sentence = input["sentence"]    
    doc = nlp(sentence)
    PARAMS = {'utterance':sentence}
    r = requests.get(url = URL, params = PARAMS)
    response = r.json()
    try:
        for prop in response["props"]:
            for span in prop["spans"]:
                if span["vn"] in {"Product", "Result", "Patient", "Theme"}:
                    text = span["text"]
                    start_pos = doc[int(span["start"]):int(span["start"]+1)].start_char
                    end_pos = doc[int(span["end"]):int(span["end"]+1)].end_char
                    vn = span["vn"]
                    tokens = text.split(" ")
                    stemmed_tokens = [ps.stem(token) for token in tokens]
                    string_1 = " ".join(stemmed_tokens)
                    for row in visual_works:
                        aat_id = row[0]
                        label = row[1]
                        if label.startswith("<"):
                            pass
                        else:
                            label_no_rounds = re.sub("\s\(.*?\)$", "", label)
                            string_2 = " ".join([ps.stem(w) for w in label_no_rounds.split(" ")])
                            if string_1 == string_2:
                                output.append([input["id"], start_pos, end_pos, token, aat_id, label, "visual_work", vn])
                    for row in buildings:
                        aat_id = row[0]
                        label = row[1]
                        if label.startswith("<"):
                            pass
                        else:
                            label_no_rounds = re.sub("\s\(.*?\)$", "", label)
                            string_2 = " ".join([ps.stem(w) for w in label_no_rounds.split(" ")])
                            if string_1 == string_2:
                                output.append([input["id"], start_pos, end_pos, token, aat_id, label, "buildings", vn])
                    for row in components:
                        aat_id = row[0]
                        label = row[1]
                        if label.startswith("<"):
                            pass
                        else:
                            label_no_rounds = re.sub("\s\(.*?\)$", "", label)
                            string_2 = " ".join([ps.stem(w) for w in label_no_rounds.split(" ")])
                            if string_1 == string_2:
                                output.append([input["id"], start_pos, end_pos, token, aat_id, label, "components", vn])
    except KeyError:
        sentence_lst = tokenize.sent_tokenize(sentence)
        for num, sm_sentence in enumerate(sentence_lst):
            doc = nlp(sm_sentence)
            PARAMS = {'utterance':sm_sentence}
            try:
                r = requests.get(url = URL, params = PARAMS)
                response = r.json()
                for prop in response["props"]:
                    for span in prop["spans"]:
                        if span["vn"] in {"Product", "Result", "Patient", "Theme"}:
                            text = span["text"]
                            if num==0:
                                start_pos = doc[int(span["start"]):int(span["start"]+1)].start_char
                                end_pos = doc[int(span["end"]):int(span["end"]+1)].end_char
                                vn = span["vn"]
                                tokens = text.split(" ")
                                for token in tokens:
                                    string_1 = ps.stem(token) 
                                    for row in visual_works:
                                        aat_id = row[0]
                                        label = row[1]
                                        if label.startswith("<"):
                                            pass
                                        else:
                                            label_no_rounds = re.sub("\s\(.*?\)$", "", label)
                                            string_2 = " ".join([ps.stem(w) for w in label_no_rounds.split(" ")])
                                            if string_1 == string_2:
                                                output.append([input["id"], start_pos, end_pos, token, aat_id, label, "visual_work", vn])
                                    for row in buildings:
                                        aat_id = row[0]
                                        label = row[1]
                                        if label.startswith("<"):
                                            pass
                                        else:
                                            label_no_rounds = re.sub("\s\(.*?\)$", "", label)
                                            string_2 = " ".join([ps.stem(w) for w in label_no_rounds.split(" ")])
                                            if string_1 == string_2:
                                                output.append([input["id"], start_pos, end_pos, token, aat_id, label, "buildings", vn])
                                    for row in components:
                                        aat_id = row[0]
                                        label = row[1]
                                        if label.startswith("<"):
                                            pass
                                        else:
                                            label_no_rounds = re.sub("\s\(.*?\)$", "", label)
                                            string_2 = " ".join([ps.stem(w) for w in label_no_rounds.split(" ")])
                                            if string_1 == string_2:
                                                output.append([input["id"], start_pos, end_pos, token, aat_id, label, "components", vn])
                            else: 
                                start_pos = doc[int(span["start"]):int(span["start"]+1)].start_char+len(sentence_lst[num-1])
                                end_pos = doc[int(span["end"]):int(span["end"]+1)].end_char+len(sentence_lst[num-1])
                                vn = span["vn"]
                                tokens = text.split(" ")
                                for token in tokens:
                                    string_1 = ps.stem(token)
                                    for row in visual_works:
                                        aat_id = row[0]
                                        label = row[1]
                                        if label.startswith("<"):
                                            pass
                                        else:
                                            label_no_rounds = re.sub("\s\(.*?\)$", "", label)
                                            string_2 = " ".join([ps.stem(w) for w in label_no_rounds.split(" ")])
                                            if string_1 == string_2:
                                                output.append([input["id"], start_pos, end_pos, token, aat_id, label, "visual_work", vn])
                                    for row in buildings:
                                        aat_id = row[0]
                                        label = row[1]
                                        if label.startswith("<"):
                                            pass
                                        else:
                                            label_no_rounds = re.sub("\s\(.*?\)$", "", label)
                                            string_2 = " ".join([ps.stem(w) for w in label_no_rounds.split(" ")])
                                            if string_1 == string_2:
                                                output.append([input["id"], start_pos, end_pos, token, aat_id, label, "buildings", vn])
                                    for row in components:
                                        aat_id = row[0]
                                        label = row[1]
                                        if label.startswith("<"):
                                            pass
                                        else:
                                            label_no_rounds = re.sub("\s\(.*?\)$", "", label)
                                            string_2 = " ".join([ps.stem(w) for w in label_no_rounds.split(" ")])
                                            if string_1 == string_2:
                                                output.append([input["id"], start_pos, end_pos, token, aat_id, label, "components", vn])
            except KeyError:
                print("error for sentence num", input["id"], num)

    
    pbar.update(1)


with open('matched_terms', 'w', encoding="utf-8") as f:
    write = csv.writer(f)
    write.writerow(fields)
    write.writerows(output)