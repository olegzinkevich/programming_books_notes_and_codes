# Sometimes your application is designed in such a way that mocking just isn’t applicable or
# won’t work. These instances most often occur when you import some library or your own
# code to perform a function but don’t want to call the real code in the test. If there is no way
# of injecting the mock, like in the previous example, then you must “patch” the code to replace
# the real instance with an instance of Mock. Once you have applied the patch, the method of
# setting up the return values is the same as in the Mock example

# As in the database example, you do not want to call these real web services in your tests,
# because while you are running the tests these services may be unavailable or a network
# outage may cause intermittent test failures that have nothing to do with the code.
# To patch a library such as this, you can use the @patch decorator, and specify the target you
# want to patch with a Mock instance

# Now if you want to test that when the method is called and you get a
# successful response from the GET request, the data is returned to you, and you need to apply
# the patch on the requests library

import unittest
from mock import Mock, patch
from ch01_intro.account_db import Account

class TestAccount(unittest.TestCase):

    @patch('ch01_intro.account_db.requests')
    def test_get_current_balance_returns_data_correctly(self,
                                                        mock_requests):
        mock_requests.get.return_value = '500'
        account = Account(Mock())
        self.assertEqual('500', account.get_current_balance('1'))

    @patch('ch01_intro.account_db.requests')

    def test_get_current_balance_returns_full_data(self,
                                                        mock_requests):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = 'Some text data'
        mock_requests.get.return_value = mock_response
        account = Account(Mock())
        self.assertEqual({'status': 200, 'data': 'Some text data'},
                         account.get_current_balance_full_data('1'))