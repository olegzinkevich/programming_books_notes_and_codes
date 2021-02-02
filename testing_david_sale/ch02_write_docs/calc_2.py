# Fortunately, you can apply some
# tricks to your doctests to reduce the amount of code you need to write in them. when you
# write your doctests you must first create an instance of the Calculate class (c = Calculate()), store this in a
# variable, and then call your method as expected. In a class with many methods, creating this
# instance every time can be tedious and take up valuable space for the doctest and code

# The doctest module allows for a context to be passed in, which is then available to all doctests in
# the file. So instead of creating the instance c = Calculate() every time, you can do this  just once in the doctest runner.

class Calculate(object):
    """Takes two integers and adds them together to produce
    the result.

    >>> c.add(1, 3)
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
            raise TypeError("Invalid type: {} and {}".
                            format(type(x), type(y)))


if __name__ == "__main__":

    import doctest
    doctest.testmod(globs={'c': Calculate()})

# run in cmd: python calc_2.py -v
