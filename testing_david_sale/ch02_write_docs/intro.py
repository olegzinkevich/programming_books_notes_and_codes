# Fortunately, there is a happy medium. Documentation doesn’t need to be in the form of a
# huge document or website. It can be a combination of short formal documents (such as a
# website or wiki page), doc strings in methods, comments in code, and most importantly, the
# code itself. If your code follows good practices such as clear naming conventions and consistent
# use of styles, you can keep comments and documentation to a minimum  By
# keeping comments to a minimum, you avoid having out-of-date documentation

# But what if you could take it a step further, and actually have living documentation that tests
# the methods they are describing? What if you then build these tests into your build suite, so
# that if that method’s behavior changes, the documentation test fails and you are forced to
# update the documentation?

# The doctest module is included in the Python standard library, so you should not need to
# install anything to start writing a doctest. Doctest approaches the testing of your methods in
# a slightly different way than the unit tests you have looked at so far. Rather than using explicit
# methods to check the return value or exception raised by your method, you essentially provide
# an example of using the method in a Python shell and its expected output. With the
# doctests in place, you can then execute them in much the same way as unit test, as part of
# your daily testing suite.

# use in shell $ python