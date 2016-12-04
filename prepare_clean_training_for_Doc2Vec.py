from __future__ import print_function
from nltk.corpus import stopwords
import os,math,random,pymysql,time,csv,numpy,re,string


# The output of clean raw data looks like this
"""
raw[0] = 
[
 '27944',
 'nike men free train versatility shoes mens black black white',
 'nike free train versatility men shoes feature wrap knit mesh upper flywire technology lace closure dual density midsole provides auxetic transition allowing foot splay throughout shoe lightweight rubber traction outsole imported color black black white material rubber lace mesh pattern solid available narrow width extended sizes',
 ''
 ]
 
 Each item in raw corresponds to an record in database. 
 1st field is id, 2nd is tile, 3rd is long description, 4th is short discription (often missing).
 """



# Connect to database
cnx_mysql = pymysql.connect(user='ext_root', password='Poln@mysql123',
                          host='45.55.62.171',
                          database='DB_Training_Data')
cur_mysql = cnx_mysql.cursor()

# Store file as a Python list
cur_mysql.execute('''
                     SELECT id, title, long_description, short_description 
                     FROM product_data_new as pdn
                  ''')

data = cur_mysql.fetchall()
try:
    print("# of data:{}".format(len(data)))
    cur_mysql.close()
except Exception as e:
    print(e.message)

# Save the Python list as a csv file
with open("product_data_new_titleAndLongShortDescription.csv", "wb") as file_out:
    writer = csv.writer(file_out)
    for line in data:
        writer.writerow(line)

# Load the csv file of raw data
with open("product_data_new_titleAndLongShortDescription.csv", "rb") as file_in:
    reader = csv.reader(file_in)
    raw = [line for line in reader]
    
############ Data Cleaning #############

# Retain only printables characters (removes non-eng chars)
printables = set(string.printable)
for item in raw:
    for i, field in enumerate(item):
        item[i] = filter(lambda x: x in printables, field)

# All to lower cases
for item in raw:
    for i, field in enumerate(item):
        item[i] = field.lower()

# Replace connectors and punctuations into space
pattern = "[\.\,\:\-\?\!\_\'\/]"
for item in raw:
    for i, field in enumerate(item):
        item[i] = re.sub(pattern, " ", field)

# Remove brackets <>, [], () and contents in them
pattern = "[\(\[\<].*?[\)\]\>]"
for item in raw:
    for i, field in enumerate(item):
        item[i] = re.sub(pattern, "", field)

# Remove all \n, \t space-like characters
pattern = "[\n\t]"
for item in raw:
    for i, field in enumerate(item):
        item[i] = re.sub(pattern, " ", field)

# Remove all numbers
pattern = "\d"
for item in raw: 
    for i, field in enumerate(item):
        if i != 0: # skip the id field
            item[i] = re.sub(pattern, "", field)

# Remove stop words
stop_words = set(stopwords.words("english"))
exclusions = {"he", "him", "his", "she", "her"}
stop_words = set(filter(lambda x: x not in exclusions, stop_words))
for item in raw:
    for i, field in enumerate(item):
            if i != 0: # skip the id field
                splited = field.split()
                item[i] = " ".join([token for token in splited \
                           if token not in stop_words])
            
# Remove any words that is less than 2 characters long
least_length = 2
for item in raw:
    for i, field in enumerate(item):
        if i != 0: # skip the id field
            splited = field.split()
            item[i] = " ".join([token for token in splited \
                       if (len(token)>=least_length or token=="he")])

# Replace all multiple spaces with one space
pattern = "\s\s+"
for item in raw:
    for i, field in enumerate(item):
        item[i] = re.sub(pattern, " ", field)

# Save the Python list as a csv file
with open("product_data_new_titleAndLongShortDescription_clean.csv"
          , "wb") as file_out:
    writer = csv.writer(file_out)
    for line in raw:
        writer.writerow(line)

# Load the csv file of raw data
with open("product_data_new_titleAndLongShortDescription_clean.csv"
          , "rb") as file_in:
    reader = csv.reader(file_in)
    clean_raw = [line for line in reader]
