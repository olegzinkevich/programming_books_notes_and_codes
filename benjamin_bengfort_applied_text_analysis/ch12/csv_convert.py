# # Converts CSV to multiple JSONS

target = "C:/Users/810004/Desktop/Amazon_sample"

import csv
import json
import os

# Open the CSV
f = open( 'C:/Users/810004/Desktop/Amazon_sample/amazon.csv', 'rU' )
# Change each fieldname to the appropriate field name. I know, so     difficult.
reader = csv.DictReader( f, fieldnames = ('id','name','asins','brand','categories','keys','manufacturer','reviews.date','reviews.dateAdded','reviews.dateSeen','reviews.didPurchase','reviews.doRecommend','reviews.id','reviews.numHelpful','reviews.rating','reviews.sourceURLs','reviews.text','reviews.title','reviews.userCity','reviews.userProvince','reviews.username' ), )
# Parse the CSV into JSON

for idx, row in enumerate(reader, 1):

    fname = str(idx) + row['id'] + '.json'
    abspath = os.path.normpath(os.path.join(target, fname))
    parent = os.path.dirname(abspath)
    if not os.path.exists(parent):
        os.makedirs(parent)
    id = row['id']
    text = row['reviews.text']
    rating = row['reviews.rating']
    dict = {'id':id, 'text':text, 'rating':rating}
    print(dict)
    print(idx)
    with open(abspath, 'w') as outfile:
        print('writing file')
        json.dump(dict, outfile)

