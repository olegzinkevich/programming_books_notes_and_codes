import unittest
from ch01_intro.calculate import Calculate



class TestCalculate(unittest.TestCase):
    # You do this in the setUp method, which is executed before
    # each test, so that you need define your instance only once and have it created before each test. the addition of the setUp method means you only
    # need to create the instance of Calculate once for it to be available to all test cases. Say your class had grown and you now had 15 test cases
    # where this is now created. What if the initializer for Calculate changed and you now
    # needed to pass in some new variables to the class? Instead of just one change in the setUp,
    # you now need to change 15 lines of code.
    def setUp(self):
        self.calc = Calculate()

    # All methods that take an optional argument, msg=None, can be provided a custom message that is displayed on failure.
    # simple assertion equal
    def test_add_method_returns_correct_result(self):
        self.assertEqual(4, self.calc.add(2,2))

    # The assert
    # Raises method, also found in the unittest package, provides you with a means to check
    # that a method raises an exception under certain circumstances The unit test method assertRaises takes three arguments. The first is the type
    # of exception you expect to be raised, in this case TypeError. The second is the method
    # under test, in this case self.calc.add that you expect to raise this exception. The final
    # value passed in is the argument to the method under test, in this case the string "Hello".
    def test_add_method_raises_typeerror_if_not_ints(self):
        self.assertRaises(TypeError, self.calc.add, "Hello","World")

#     assertRaises method provides the capability
# to use it as a context manager. This means you can execute any code you like within the context
# of assertRaises, and if the exception is raised it will be caught and your test will pass
# as expected.
    def test_assert_raises(self):
        with self.assertRaises(AttributeError):
            [].get()

#     or
    def test_assert_raises(self):
        self.assertRaises(ValueError, int, "a")

# assertEqual(x, y, msg=None)
# This method checks to see whether argument x equals argument y. Under the covers, this
# method is performing the check using the == definition for the objects.
    def test_assert_equal(self):
        self.assertEqual(1, 1)

#     assertAlmostEqual(x, y, places=None, msg=None, delta=None)
# On first glance, this method may seem a little strange but in context becomes useful. The
# method is basically useful around testing calculations when you want a result to be within a
# certain amount of places to the expected, or within a certain delta.
    def test_assert_almost_equal_places(self):
        self.assertAlmostEqual(1, 1.00001, delta=0.5)

# assertDictContainsSubset(expected, actual, msg=None)
# Use this method to check whether actual contains expected. It’s useful for checking that
#     part of a dictionary is present in the result, when you are expecting other things to be there
# also. For example, a large dictionary may be returned and you need to test that only a couple
# of entries are present.
    def test_assert_dict_contains_subset(self):
        expected = {'a': 'b'}
        actual = {'a': 'b', 'c': 'd', 'e': 'f'}
        self.assertDictContainsSubset(expected, actual)

# assertDictEqual(d1, d2, msg=None)
# This method asserts that two dictionaries contain exactly the same key value pairs. For this
# test to pass, the two dictionaries must be exactly the same, but not necessarily in the same
# order.
    def test_assert_dict_equal(self):
        expected = {'a': 'b', 'c': 'd'}
        actual = {'c': 'd', 'a': 'b'}
        self.assertDictEqual(expected, actual)

# assertTrue(expr, msg=None)
#     Value Truth
# 0 False
# 1 True
# -1 True
# “” False
# “Hello, World!” True
# None False
    def test_assert_true(self):
        self.assertTrue(1)
        self.assertTrue("Hello, World")

# assertFalse(expr, msg=None)
# This method is the inverse of assertTrue and is used for checking whether the expression
# or result under the test is False.
    def test_assert_false(self):
        self.assertFalse(0)
        self.assertFalse("")

# assertGreater(a, b, msg=None)
# This method allows you to check whether one value is greater than the other. It is essentially
# a helper method that wraps up the use of assertTrue on the expression a > b. It displays
# a helpful message by default when the value is not greater.

    def test_assert_greater(self):
        self.assertGreater(2, 1)

# assertGreaterEqual(a, b, msg=None)
# You use this method to check whether one value is greater than or equal to another value.
# Essentially, this wrapper is asserting True on a >= b. The assertion also gives a nicer message
# if the expectation is not met.

    def test_assert_greater_equal(self):
        self.assertGreaterEqual(2, 2)

    # assertIn(member, container, msg=None)
    # With this method, you can check whether a value is in a container (hashable) such as a list or
    # tuple. This method is useful when you don’t care what the other values are, you just wish to
    # check that a certain value(s) is in the container.

    def test_assert_in(self):
        self.assertIn(1, [1,2,3,4,5])

    # assertIs(expr1, expr2)
    # Use this method to check that expr1 and expr2 are identical. That is to say they are the
    # same object. For example, the python code [] is [] would return False, as the creation
    # of each list is a separate object.

    def test_assert_is(self):
        self.assertIs("a", "a")

    # assertIsInstance(obj, class, msg=None)
    # This method asserts that an object is an instance of a specified class. This is useful for checking
    # that the return type of your method is as expected (for instance, if you wish to check that
    # a value is a type of int):

    def test_assert_is_instance(self):
        self.assertIsInstance(1, int)

    # assertNotIsInstance(obj, class, msg=None)
    # This reverse of assertIsInstance provides an easy way to assert that the object is not a
    # type of the class.

    def test_assert_is_not_instance(self):
        self.assertNotIsInstance(1, str)

    # assertIsNone(obj, msg=None)
    # Use this to easily check if a value is None. This method provides a useful standard message if
    # not None.
    def test_assert_is_none(self):
        self.assertIsNone(None)

    # assertIsNot(expr1, expr2, msg=None)
    # Using this method, you can check that expr1 is not the same as expr2. This is to say that
    # expr1 is not the same object as expr2.
    #
    def test_assert_is_not(self):
        self.assertIsNot([], [])

    # assertIsNotNone(obj, msg=None)
    # This method checks that the value provided is not None. This is useful for checking that your
    # method returns an actual value, rather than nothing.

    def test_assert_is_not_none(self):
        self.assertIsNotNone(1)

    # assertLess(a, b, msg=None)
    # This method checks that the value a is less than the value b. This is a wrapper method for
    #     assertTrue on a < b.

    def test_assert_less(self):
        self.assertLess(1, 2)

    # assertLessEqual(a, b, msg=None)
    # This method checks that the value a is less than or equal to the value b. This is a wrapper
    # method for assertTrue on a <= b.

    def test_assert_less_equal(self):
        self.assertLessEqual(2, 2)

    # assertItemsEqual(a, b, msg=None)
    # This assertion is perfect for testing whether two lists are equal. Lists are unordered; therefore,
    # assertEqual on a list can produce intermittent failing tests as the order of the lists
    # may change when running the tests. This can produce a failing test when in fact the two lists
    # have the same contents and are equal.

    def test_assert_items_equal(self):
        self.assertItemsEqual([1,2,3], [3,1,2])

    #     assertRaises(excClass, callableObj, *args, **kwargs, msg=None)
    # This assertion is used to check that under certain conditions exceptions are raised. You pass
    # in the exception you expect, the callable that will raise the exception and any arguments to
    # that callable. In the earlier example, this pops the first item from an empty list and results in
    # an IndexError.

    def test_assert_raises(self):
        self.assertRaises(IndexError, [].pop, 0)

if __name__ == '__main__':
    unittest.main()