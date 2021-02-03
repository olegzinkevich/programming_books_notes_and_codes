import json



DATA = "feed.json"

blog_data = json.loads(open(DATA).read())

#extract contents
q = blog_data['title']['content'][0]
print(q)