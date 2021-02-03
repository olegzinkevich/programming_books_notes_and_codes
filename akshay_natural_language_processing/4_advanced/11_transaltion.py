# pip install goslate
import goslate

text = "Bonjour le monde"
gs = goslate.Goslate()
translatedText = gs.translate(text,'en')

print(translatedText)