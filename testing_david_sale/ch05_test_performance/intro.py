# How will your application
# perform when you scale up from only a couple of requests per second to thousands,
# hundreds of thousands, and possibly millions? This is where performance testing comes into
# play and in particular the use of the JMeter tool. JMeter allows you to mimic some of the
# real-world conditions you expect your application to face to reveal the kind of response times
# your application will deliver by generating load on your application. As you scale up the
# amount of requests per second, is there a tipping point where suddenly the response times
# increase and performance suf fers?

# Performance testing also helps you plan various parts of your application and can influence
# design decisions. How does your application perform on a cloud environment versus a noncloud
# environment? Do you need the capability to scale up and down quickly? How many
# instances of your application do you need to run to cope with an average amount of traffic? All
# these questions raise issues around costs and procurement. Your project’s budget may allow only
# a certain amount of space in the cloud. You may have software or hardware limits and so your
# application must be as performative as possible to make best use of your available resources.
# Some problems cannot be overcome simply by throwing as much hardware at them as possible.
# Finding performance issues such as blocking I/O calls, lock contention, and CPU intensive operations
# allows you to refactor code and perhaps find a better alternative than the initial solution

# we use JMeter to generate the load on your application.

# Although JMeter is written in Java, you don’t need to know
# any Java code. You simply need to learn about the fields and options the tool provides to be
# able to build your test plan. You can then save your test plans, most likely in a source control
# repository, and reuse them to test your application as it is developed.

# You can find the documentation for the tool on the JMeter website at
# https://jmeter.apache.org.

# Ubuntu, you can install JMeter using your package manager:
# $ sudo apt-get install jmeter

# cProfile

# A lower level, perhaps rudimentary way of getting performance statistics from your code is
# through the use of code profiling tools. Python actually ships a code profiler as part of the
# standard library known as cProfile. With cProfile already available to you, you can dive
# straight into profiling your code and getting some stats. By using these tools, you can glean
# useful performance information, such as which methods and classes have the most calls. You
# can also get information such as the time spent in those methods, so by using these stats you
# can determine problem areas in your code that you could perhaps refactor and break down
# into more efficient calls
