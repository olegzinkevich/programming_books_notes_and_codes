
class Account(object):

    def __init__(self, account_number, balance):

        self.account_number = account_number
        self.balance = balance

# You can create individual Account objects, but they are not stored anywhere
# in which they can be retrieved easily. It looks like a Bank object is needed to store all
# the Accounts. The simplest form the Bank object can take is a dictionary, where the key is the
# account_number and the value is the balance.

class Bank(object):

    def __init__(self):
        self.accounts = {}

    def add_account(self, account):
        self.accounts[account.account_number] = account.balance

    def get_account_balance(self, account_number):
        return self.accounts.get(account_number)