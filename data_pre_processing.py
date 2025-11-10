import pandas as pd
import ast
import re
import csv

# Read input CSVs
df_train = pd.read_csv('data/train.csv')
dialogs_train = df_train['dialog'].tolist()

df_validation = pd.read_csv('data/validation.csv')
dialogs_validation = df_validation['dialog'].tolist()

df_test = pd.read_csv('data/test.csv')
dialogs_test = df_test['dialog'].tolist()

#lists of prompt and responses
prompt_response_train = []
prompt_response_validation = []
prompt_response_test = []

#splitting sentences
def split_sentences(single_string_list):
    text = single_string_list[0]
    sentences = re.split(r'(?<=[.!?]) +', text)
    return [s.strip() for s in sentences if s.strip()]

# Extract prompt–response pairs
for convo in dialogs_train:
    convo_list = ast.literal_eval(convo)
    sentences = split_sentences(convo_list)
    for i in range(len(sentences) - 1):
        prompt_response_train.append([sentences[i], sentences[i + 1]])

for convo in dialogs_validation:
    convo_list = ast.literal_eval(convo)
    sentences = split_sentences(convo_list)
    for i in range(len(sentences) - 1):
        prompt_response_validation.append([sentences[i], sentences[i + 1]])

for convo in dialogs_test:
    convo_list = ast.literal_eval(convo)
    sentences = split_sentences(convo_list)
    for i in range(len(sentences) - 1):
        prompt_response_test.append([sentences[i], sentences[i + 1]])

# Save pairs to CSV
with open('train_processed.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f, quoting=csv.QUOTE_ALL)
    writer.writerow(['prompt', 'response'])
    writer.writerows(prompt_response_train)

with open('validation_processed.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f, quoting=csv.QUOTE_ALL)
    writer.writerow(['prompt', 'response'])
    writer.writerows(prompt_response_validation)   

with open('test_processed.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f, quoting=csv.QUOTE_ALL)
    writer.writerow(['prompt', 'response'])
    writer.writerows(prompt_response_test)

# Build clean sequential text corpus
all_sentences = []
for p, r in prompt_response_train:
    if not all_sentences or p != all_sentences[-1]:
        all_sentences.append(p)
    all_sentences.append(r)

# Save final corpus
with open("text_corpus.txt", "w", encoding="utf-8") as f:
    for sentence in all_sentences:
        f.write(sentence.strip() + "\n")
