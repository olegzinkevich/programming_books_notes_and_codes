Parallelism (parallel or distributed computation) has two primary forms. Task parallelism
means that different, independent operations run simultaneously on the same
data. Data parallelism implies that the same operation is being applied to many different
inputs simultaneously. Both task and data parallelism are often used to accelerate
computation from its sequential form (one operation at a time, one after the other)
with the intent that the computation becomes faster.

# When a Python program is run, the operating system executes the code as a
process. Processes are independent, and in order to share information between them,
some external mechanism is required (such as writing to disk or a database, or using
a network connection).

A single process might then spawn several threads of execution. A thread is the smallest
unit of work for a CPU scheduler and represents a sequence of instructions that
must be performed by the processor. Whereas the operating system manages processes,
the program itself manages its own threads, and data inherent to a certain process
can be shared among that process’s threads without any outside communication.

# gil

Unfortunately, Python cannot take advantage
of multiple cores due to the Global Interpreter Lock, or GIL, which ensures that
Python bytecode is interpreted and executed in a safe manner. Therefore, whereas a
Go program might achieve a CPU utilization of 200% in a dual-core computer,
Python will only ever be able to use at most 100% of a single core.

There are two primary modules for parallelism within Python: threading and multi
processing. Both modules have a similar basic API (meaning that you can easily
switch between the two if needed), but the underlying parallel architecture is fundamentally
different.

To achieve parallelism in Python, the multiprocessing library is required. The multiprocessing
module creates additional child processes with the same code as the parent
process by either forking the parent on Unix systems (the OS snapshots the
currently running program into a new process) or by spawning on Windows (a new
Python interpreter is run with the shared code). Each process runs its own Python
interpreter and has its own GIL, each of which can utilize 100% of a CPU. Therefore,
if you have a quad-core processor and run four multiprocesses, it is possible to take
advantage of 400% of your CPU.


The multiprocessing architecture in Figure 11-1 shows the typical structure of a parallel
Python program. It consists of a parent (or main) program and multiple child
processes (usually one per core, though more is possible). The parent program schedules
work (provides input) for the children and consumes results (gathers output).
Data is passed to and from children and the parent using the pickle module.

Workflow - look parallel.png

In Figure 11-1, two different vectorization tasks are run in parallel and the main process
waits for them all to complete before moving on to a fitting task (e.g., two models
on each of the different vectorization methods) that also runs in parallel. Forking
causes multiple child processes to be instantiated, whereas joining causes child processes
to be ended, and control is passed back to the primary process. For instance,
during the first vectorize task, there are three processes: the main process and the
child processes A and B. When vectorization is complete, the child processes end and
are joined back into the main processes.



