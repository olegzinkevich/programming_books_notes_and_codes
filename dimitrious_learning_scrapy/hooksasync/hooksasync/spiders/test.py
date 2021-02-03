# hookasync - a project of a spider with all avalable signals.
# Signals provide a mechanism to add callbacks to events that happen in the system, # such as when a spider opens, or when an item gets scraped. You can hook to them # using the crawler.signals.connect() method (an example of using it can be # found in the next section). There are just 11 of them and maybe the easiest way # to understand them is to see them in action.

#  look at extensions.py

from scrapy.spider import Spider
from scrapy.item import Item, Field


class HooksasyncItem(Item):
    name = Field()


class TestSpider(Spider):
    name = "test"
    allowed_domains = ["example.com"]
    start_urls = ('http://www.example.com',)

    def parse(self, response):
        for i in range(2):
            item = HooksasyncItem()
            item['name'] = "Hello %d" % i
            yield item
        raise Exception("dead")
