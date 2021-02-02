# Running doctests by executing the Python file they are in is an okay process for testing in
#     this style, but what if you start to expand your doctests to many classes and methods? Also,
# running the tests as part of a build process will be difficult to manage should you need to
# maintain a list of which files to execute that contain doctests.
# Fortunately, you can make use of some tight integration with the nosetest runner, which was
# introduced earlier in the book. Nosetest provides a way of appending a flag to your normal
# nosetest command, which searches through the files for doctests and executes them

# You enable the nose’s doctest support by running the nosetests command with the flag
# –with-doctest. Nosetests then searches all the files for doctests and executes them
# alongside your unit tests

# nosetests --with-doctest -v