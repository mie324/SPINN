import xml.etree.ElementTree as ET
import pandas as pd
tree = ET.parse('./semeval2017_task7/data/test/subtask1-homographic-test.xml')
root = tree.getroot()
sentence_list = []
for i in range(2250):
    sentence_length = len(root[i])
    word_list = []
    for j in range(sentence_length):
        word = root[i][j].text
        word_list.append(word)
    sentence_str = ' '.join(word_list)
    sentence_list.append(sentence_str)
df = pd.DataFrame(sentence_list,columns = ['text'])
df.to_csv('sentences.csv',index=True)
print(df)
