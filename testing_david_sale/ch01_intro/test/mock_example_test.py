import unittest
from mock import Mock

# In this example, you can create an instance of the mock named my_mock, add the
# my_method to it, and state that when it is called, it should return the string "hello". When you write tests, you may discover that your code interacts with other systems that are
# assumed to be tested and fully functional, such as call a database or a web service. In these
# instances, you don’t want to make those calls for real in your test—for a number of reasons.
# For example, the database or web service may be down and you don’t want that to cause a
# failing test as part of your build process, which would produce a false positive.
class TestMocking(unittest.TestCase):

    def test_mock_method_returns(self):
        # You simply need to import the Mock class and then create an
        # instance of it. You can then attach methods to the mock that you want to return some value.
        # Create a test file called mock_example_test.py and use the following code
        my_mock = Mock()
        # In this example, you can create an instance of the mock named my_mock, add the
        # my_method to it, and state that when it is called, it should return the string "hello".
        my_mock.my_method.return_value = "hello"
        self.assertEqual("hello", my_mock.my_method())

if __name__ == '__main__':
    unittest.main()
#
