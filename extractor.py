from transformers import pipeline
from gliner import GLiNER

import json

# set title or use first line
f = open("test.txt", "r")
title = f.readline()

sections = f.read().split(title)

resultJson = {
    "title": title,
    "entities": [],
    "summary": "",
    "outline": {}
}

summarizer = pipeline("summarization", model="Falconsai/text_summarization")
summaries = []
entities = []

# identify entities and add to a list
model = GLiNER.from_pretrained("urchade/gliner_multi_pii-v1")

labels = ["account number", "date of birth", "driver license", "person", "full address", "email", "passport number", "Social Security Number", "phone number"]

for section in sections:

    # build entities
    sectionEntities = model.predict_entities(section, labels)
    for entity in sectionEntities: 
        entities.append(entity["text"])

    # keep track of summaries
    summary = summarizer(section, min_length=30, do_sample=False)
    summaries.append(summary[0].get("summary_text"))


# create summary of summaries for the entire document
joinedSummaries = ' Next Entry: '.join(summaries)
resultJson["summary"] = summarizer(joinedSummaries, min_length=30, do_sample=False)[0].get("summary_text")
# add entities
resultJson["entities"] = entities

print(json.dumps(resultJson))