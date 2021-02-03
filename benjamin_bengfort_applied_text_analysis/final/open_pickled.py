import pickle

pickle_in = open(r"C:\Users\810004\Desktop\Html_corpus\raw_json_pickled\business\56d7aebfc1808109b3876702.pickle", "rb")
example_dict = pickle.load(pickle_in)

print(example_dict)

# todo: проверить разные pickles , после ручной processing и после прделагаемых в книге. Совпадают ли.




