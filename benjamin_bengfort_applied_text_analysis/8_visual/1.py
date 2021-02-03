we performed grammar-based feature extraction, aiming to identify significant
patterns of tokens across many documents. In practice, we will have a much
easier time steering this phase of the workflow if we can visually explore the frequency
of combinations of tokens as a function of time.

Let’s assume that the corpus data has been formatted as a dictionary where keys are
corpus tokens and the values are (token count, document datetime stamp) tuples.
Assume that we also have a comma-separated list, terms, with strings that correspond
to the n-grams we would like to plot as a time series.

In order to explore n-grams over time, we begin by initializing a Matplotlib figure
and axes, with the width and height dimensions specified in inches. For each term in
our term list, we will plot the count of the target n-gram as the x-value and the datetime
stamp of the document in which the term appeared as the y-value

