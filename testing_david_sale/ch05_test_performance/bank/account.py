
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

def main():
    account = Account('1111', 50)
    bank = Bank()
    bank.add_account(account)
    for x in range(1,1000):
        print(x*(x**2))

if __name__ == '__main__':
    # cProfile

    # A lower level, perhaps rudimentary way of getting performance statistics from your code is
    # through the use of code profiling tools. Python actually ships a code profiler as part of the
    # standard library known as cProfile. With cProfile already available to you, you can dive
    # straight into profiling your code and getting some stats. By using these tools, you can glean
    # useful performance information, such as which methods and classes have the most calls. You
    # can also get information such as the time spent in those methods, so by using these stats you
    # can determine problem areas in your code that you could perhaps refactor and break down
    # into more efficient calls
    import cProfile

    cProfile.run('main()', sort='time')

    # If you don’t have a main method such as the preceding example to start your application, to run
    # the profile you pass the cProfile library as a parameter to Python when calling your Python script:
    # $ python -m cProfile bank_app.py
