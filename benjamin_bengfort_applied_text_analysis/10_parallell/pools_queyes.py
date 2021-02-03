In the previous section we looked at how to use the multiprocessing.Process object
to run individual tasks in parallel. The Process object makes it easy to define individual
functions to run independently, closing the function when it is complete. More
advanced usage might employ subclasses of the Process object, each of which must
implement a run() method that defines their behavior.
In larger architectures this allows easier management of individual processes and
their arguments (e.g., naming them independently or managing database connections
or other per-process attributes) and is generally used for task parallel execution. With
task parallelism, each task has independent input and output (or the output of one
task may be the input to another task). In contrast, data parallelism requires the same
task to be mapped to multiple inputs. Because the input is independent, each task can
be applied in parallel.  For data parallelism, the multiprocessing library provides pools and queues


A common combination of both data and task parallelism is to
have two data parallel tasks; the first maps an operation to many
data inputs and the second reduces the map operation to a set of
aggregations. This style of parallel computing has been made very
popular by Hadoop and Spark, which we will discuss in the next
section.

Larger workflows can be described as a directed acyclic graph (DAG), where a series
of parallel steps is executed with synchronization points in between. A synchronization
point ensures that all parts of the processing have completed or caught up before
execution continues. Data is also generally exchanged at synchronization points, sent
out to parallel tasks from the main task, or retrieved by the main task.

Probably the most common and time-consuming task applied to a corpus, however,
is preprocessing the corpus from raw text into a computable format. Preprocessing,
discussed in Chapter 3, takes a document and converts it into a standard data structure:
a list of paragraphs that are lists of sentences, which in turn are lists of (token,
part of speech) tuples. The final result of preprocessing is usually saving the document
as a pickle, which is both usually more compact than the original document as
well as easily loaded into Python for further processing.