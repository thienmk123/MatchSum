from summarizer.sbert import SBertSummarizer
import glob
import json

model = SBertSummarizer("vinai/phobert-base")
list_path_train = glob.glob("vietnews/data/train_tokenized/*.txt.seg")

train_set = []
for i in list_path_train:
    with open(i, "r") as f:
        raw_text = f.readlines()
        text = raw_text[4:]
        if "\n" in text:
            index = text.index("\n")
            text = raw_text[4:index]
        list_text = text
        text = " ".join(text).replace("\n", " ")
        sample = {
            "text": text,
            "list_text": list_text,
            "summary": raw_text[2].replace("\n", "")
        }
        train_set.append(sample)

train_dataset = []
train_ids = []
for i in train_set:
    result = model(i['text'], num_sentences=5)
    result_split = result.split(" . ")
    ids = []
    for id, sent in enumerate(i['list_text']):
        for sent_result in result_split:
            if sent_result.strip() in sent:
                ids.append(id)
    ids = list(set(ids))
    ids.sort()
    if len(ids) != 0:
        train_dataset.append({
            "text": i["text"],
            "summary": i["summary"]
        })
        train_ids.append({
            "sent_id": ids
        })
with open("preprocess/train_set.jsonl", "w") as f:
    for entry in train_dataset:
        json.dump(entry, f, ensure_ascii=False)
        f.write("\n")

with open("preprocess/index.jsonl", "w") as f:
    for entry in train_ids:
        json.dump(entry, f)
        f.write("\n")



