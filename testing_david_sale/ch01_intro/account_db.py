# The class has just one method that returns the data obtained from the database related to the
# provided ID number
import requests


class Account(object):

    def __init__(self, data_interface):
        self.di = data_interface

    def get_account(self, id_num):

        try:
            result = self.di.get(id_num)

        except ConnectionError:
            result = "Connection error occurred. Try Again."

        return result

    # you know that when you perform a GET request such as this, the requests
    # library returns an object with certain properties, such as status_code and text among
    # others
    def get_current_balance(self, id_num):

        return requests.get("http://some-account-uri/"+id_num)

    def get_current_balance_full_data(self, id_num):

        response = requests.get("http://some-account-uri/"+id_num)
        return {'status': response.status_code,
                'data': response.text}