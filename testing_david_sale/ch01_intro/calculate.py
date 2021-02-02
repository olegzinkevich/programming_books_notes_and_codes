class Calculate(object):

    def add(self, x, y):

        if type(x) == int and type(y) == int:
            return x + y
        else:
            raise TypeError("Invalid type: {} and {}".format(type(x), type(y)))

# #pragma: no cover
# the coverage lib report feature to remove lines that you know need not be tested. In this case execution of the program. Adding the comment #pragma: no cover tells the coverage tool to ignore the code on
# that line. If it is placed at class level or, in this case, on an if statement, the contents of that
# code is also removed.

if __name__ == '__main__': #pragma: no cover

    calc = Calculate()
    result = calc.add(2, 2)
    print(result)