# example of docstring with use cases

# Notice how you create an instance of the class Calculate in docstring, to then be able to call your method
# and show the expected output. This replicates exactly what you would need to do in the Python
# shell, so if you are ever unsure of what to place in your doctest, simply try it out in the shell first.

# You can also test the Traceback and exceptions - see c.add(1.0, 1.0)

class Calculate(object):
    """Takes two integers and adds them together to produce
    the result.

    >>> c = Calculate()
    >>> c.add(1, 1)
    2

    >>> c.add(1.0, 1.0)
    Traceback (most recent call last):
    ...
    TypeError: Invalid type: <class 'float'> and <class 'float'>

    >>> c.add(25, 125)
    150
    """

    def add(self, x, y):

        if type(x) == int and type(y) == int:
            return x + y
        else:
            raise TypeError("Invalid type: {} and {}".format(type(x), type(y)))


if __name__ == '__main__':

#     To enable it to run, you must first
# add a block of code that will execute when you run the Python file containing this class.
    import doctest
    doctest.testmod()

# to run doctest
# go to cmd and cd to the proper dir, then:

# python calc.py -v

# -v = verbose the steps of the test

# Now when you execute the file in the shell, you should receive no output. This is a little counterintuitive,
# but it means that the test passed and the method behaves as expected.
