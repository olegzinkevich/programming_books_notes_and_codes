# Loading Yellowbrick Datasets
# How to load Yellowbrick datasets:
# Yellowbrick provides several datasets wrangled from the UCI Machine Learning
# Repository. To download the data, clone the Yellowbrick library and run the download
# as follows:

$ git clone https://github.com/DistrictDataLabs/yellowbrick.git
$ cd yellowbrick/examples
$ python -m yellowbrick.download

# скачает датасеты в директорию data, затем ее можно переместить в рабочую директорию и использовать, загружать оттуда.

# Note that this will create a directory called data inside the tellobrick 'examples' directoy that contains subdirectories with the given data. Once downloaded, use the sklearn.datasets.base.Bunch object to load the corpus into features and target attributes, respectively, similarly to how Scikit-Learn’s toy datasets are structured:

look next - fredist.py