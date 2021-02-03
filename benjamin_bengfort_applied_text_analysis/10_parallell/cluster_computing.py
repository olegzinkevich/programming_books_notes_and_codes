# Cluster computing concerns the coordination of many individual machines connected
# together by a network in a consistent and fault-tolerant manner—for example, if a
# single machine fails (which becomes more likely when there are many machines), the
# entire cluster does not fail. Unlike the multiprocessing context, there is no single
# operating system scheduling access to resources and data. Instead, a framework is
# required to manage distributed data storage, the storage and replication of data across
# many nodes, and distributed computation, the coordination of computation tasks
# across networked computers.

# While beyond the scope of this book, distributed data storage is a
# necessary preliminary step before any cluster computation can take
# place if you want the cluster computation to happen reliably. Generally,
# cluster filesystems like HDFS or S3 and databases like Cassandra
# and HBase are used to manage data on disk.

Spark is an execution engine for distributed programs whose primary advantage is
support for in-memory computing. Because Spark applications can be written
quickly in Java, Scala, Python, and R it has become synonymous with Big Data science.
Several libraries built on top of Spark such as Spark SQL and DataFrames,
MLlib, and GraphX mean that data scientists used to local computing in notebooks
with these tools feel comfortable very quickly in the cluster context. Spark has
allowed applications to be developed upon datasets previously inaccessible to
machine learning due to their scope or size; a category that many text corpora fall
into. In fact, cluster computing frameworks were originally developed to handle text
data scraped from the web.

Spark can run in two modes: client mode and cluster mode. In cluster mode, a job is
submitted to the cluster, which then computes independently. In client mode, a local
client connects to the cluster in an interactive fashion; jobs are sent to the cluster and
the client waits until the job is complete and data is returned. This makes it possible
to interact with the cluster using PySpark, an interactive interpreter similar to the
Python shell, or in a Juypter notebook. For dynamic analysis, client mode is perfect
for quickly getting answers to questions on smaller datasets and corpora.