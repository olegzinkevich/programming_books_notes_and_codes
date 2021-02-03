# -*- coding: utf-8 -*-

import csv

import scrapy
from scrapy.http import Request
from scrapy.loader import ItemLoader
from scrapy.item import Item, Field


class FromcsvSpider(scrapy.Spider):
    name = "fromcsv"

    def start_requests(self):
        #  the getattr() Python method: getattr(self, 'variable', 'default').The getattr() method returns the value of the named attribute of an object. If not found, it returns the default value provided to the function.
        with open(getattr(self, "file", "C:/Users/810004/PycharmProjects/Linguistic_parser/venv1/drafts/Dimitrious_learning_scrapy/generic/generic/spiders/todo.csv"), "rU") as f:
            reader = csv.DictReader(f)
            for line in reader:
                request = Request(line.pop('url'))
                request.meta['fields'] = line
                yield request

    def parse(self, response):
        # Since we don't define project-wide Items for this project, we have to provide one to ItemLoader manually as follows:
        item = Item()
        l = ItemLoader(item=item, response=response)
        for name, xpath in response.meta['fields'].items():
            if xpath:
                #  We also add fields dynamically using the fields member variable of Item. To add a new field dynamically and have it populated by our ItemLoader, all we have to do is the following:
                item.fields[name] = Field()
                l.add_xpath(name, xpath)

        return l.load_item()
