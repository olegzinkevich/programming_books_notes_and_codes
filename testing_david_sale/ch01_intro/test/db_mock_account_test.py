#     You may now wonder how this is useful when testing your application. Suppose you have a
# program that looks up accounts from a database. If that account class is initialized using a
# data_interface class to call a database for the account information, then instead of providing
# a real data_interface you can instead mock the data_interface and provide
# the methods and return values you need for your test. Because the data_interface class
# is a whole other class with set responsibilities, the testing for this class should be assumed as handled elsewhere.



# This full control over the code means you remove this reliance on the real database, and bring
# elements under your control to ensure the code is tested to your standards. Testing in this
# level of isolation is useful later down the line in performance testing, where you want to limit
# or eradicate external factors so that you can test just your code in isolation, providing a true
# indication of the performance of your code

# Now write a test to check that the data is returned correctly for ID 1
#     given the data that you set up in the mock data_interface. Create a test file called
# account_test.py and try the following code:

from ch01_intro.account_db import Account
import unittest
from mock import Mock

class TestAccount(unittest.TestCase):

    def test_account_returns_data_for_id_1(self):

        account_data = {"id": "1", "name": "test"}
        mock_data_interface = Mock()
        mock_data_interface.get.return_value = account_data
        account = Account(mock_data_interface)
        self.assertDictEqual(account_data, account.get_account(1))
#
    #         You can then test that this code handles the exception and returns the correct error message
# by mocking the raising of the exception when
    def test_account_when_connect_exception_raised(self):
        mock_data_interface = Mock()
        mock_data_interface.get.side_effect = ConnectionError()
        account = Account(mock_data_interface)
        self.assertEqual("Connection error occurred. Try Again.",
                         account.get_account(1))


if __name__ == '__main__':
    unittest.main()