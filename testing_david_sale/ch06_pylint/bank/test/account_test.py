import unittest
from ch07_paver_jenkins.bank.account import Account, Bank

class TestAccount(unittest.TestCase):

    def test_account_object_can_be_created(self):

        account = Account("001", 50)

    def test_account_object_returns_current_balance(self):

        account = Account("001", 50)
        self.assertEqual(account.account_number, "001")
        self.assertEqual(account.balance, 50)


class BankTest(unittest.TestCase):

    def test_bank_is_initially_empty(self):
        # Here you are creating an instance of the Bank class, and in the first test, checking that the
        # accounts dictionary is initialized as empty, creating two accounts, adding them to the bank,
        # and checking that the accounts dictionary does indeed now contain the two items you have
        # added.
        bank = Bank()
        self.assertEqual({}, bank.accounts)
        self.assertEqual(len(bank.accounts), 0)

    def test_add_account(self):
        bank = Bank()
        account_1 = Account('001', 50)
        account_2 = Account('002', 100)
        bank.add_account(account_1)
        bank.add_account(account_2)
        self.assertEqual(len(bank.accounts), 2)

    def test_get_account_balance(self):
        bank = Bank()
        # You then check that the balance for
        # account number 001 is as expected.
        account_1 = Account('001', 50)
        bank.add_account(account_1)
        self.assertEqual(bank.get_account_balance('001'), 50)

# Now that you have your Account class, you can begin to add some of the functionality specified
# in the business requirement. You need to be able to uniquely identify a bank account and retrieve
# its balance. Bank accounts often have an account number to identify them. Add the test that will
# drive the creation of a constructor for the Account class. The constructor will take an account
# number and initial balance and stores that information on the object

if __name__ == '__main__':
    unittest.main()