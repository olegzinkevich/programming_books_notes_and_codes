# скачивает html в json, для дальнейшего чтения и обработки через HTMLcorpusReader

url = "https://bbengfort.github.io/snippets/2016/04/12/nltk-corpus-reader.html"
url_id = 1

import requests
import json

htmlContent = requests.get(url, verify=False)
data = htmlContent.text

# converts the raw HTML content into a JSON string representation
# jsonD = json.dumps(htmlContent.text)

# делаем заготовку финального json
ContentHTML = {
    'url': str(url),
    'uid': str(url_id),
    'page_content': htmlContent.text,
    'date': '10-12-2018'
}
print(ContentHTML)

with open('html.json', 'w') as outfile:
    json.dump(ContentHTML, outfile)

# jsonL = json.loads(jsonD)

