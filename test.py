doc_names = {'About_YSA': 7, 'Definition_Of_Homeless': 5}

file_names = []

for doc, length in doc_names.items():
    file_names = file_names + ([f'YSA_TXTS/{doc}/{doc}_{i}.txt' for i in range(length)])

print(file_names)