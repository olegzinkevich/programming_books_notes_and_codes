# -*- coding: utf-8 -*-
import scrapy
from properties.items import PropertiesItem
from scrapy.loader import ItemLoader
from scrapy.loader.processors import MapCompose, Join
from urllib.parse import urljoin
import datetime
import socket

# a BasicSpider class that extends scrapy.Spider. By 'extends' we mean that despite the fact that we didn't write any code, this class already "inherits" quite some functionality from the Scrapy ramework Spider class.

# to start a spider use in cmd: scrapy crawl basic

class BasicSpider(scrapy.Spider):
    name = 'basic'
    allowed_domains = ['gumtree']
    #  Since that's a tuple, we can hardcode multiple URLs, for example:
    # start_urls = (
    #     'file:///C:/Users/810004/Desktop/gumtree.html',
    #     'file:///C:/Users/810004/Desktop/gumtree2.html',
    # )

    #  or use  file to get urls from it:
    start_urls = [i.strip() for i in open('C:/Users/810004/PycharmProjects/Linguistic_parser/venv1/drafts/Dimitrious_learning_scrapy/properties/properties/spiders/todo.urls.txt').readlines()]

    def parse(self, response):
        #  adding contract (looks like docstring) - to run tests:

        """ This function parses a property page.
        @url file:///C:/Users/810004/Desktop/gumtree.html
        @returns items 1
        @scrapes title price description address image_urls
        @scrapes url project spider server date
        """

        # # we add an item = PropertiesItem() statement which creates a new item, and then we can assign expressions to its fields as follows: import it from items.py
        # item = PropertiesItem()
        #
        # # we will use spider's predefined method log() to output everything
        # item['title'] = response.xpath('//*[@id="ad-title"][1]/text()').extract()
        # item['price'] = response.xpath('//*[@class="ad-price txt-xlarge txt-emphasis inline-block"][1]/text()').re('[.0-9]+')
        # item['description'] = response.xpath('//*[@itemprop="description"][1]/text()').extract()
        # item['address'] = response.xpath('//*[@itemprop="address"][1]/text()').extract()
        # item['image_urls'] = response.xpath('//*[@id="image-gallery"]/div/div/div[1]/ul/li[1]/img/@src').extract()
        # return item

        # ItemLoaders provide many interesting ways of combining data, formatting them, and cleaning them up. Note that they are very actively developed so check the excellent documentation in http://doc.scrapy.org/en/latest/topics/loaders.html. ItemLoaders pass values from
# XPath/CSS expressions through different processor classes. Processors are fast yet
# simple functions. An example of a processor is Join() or MapCompose. This processor, assuming that you have selected multiple paragraphs with some XPath expression like //p, will join their text together in a single entry.
        l = ItemLoader(item=PropertiesItem(), response=response)

        l.add_xpath('title', '//*[@id="ad-title"][1]/text()', MapCompose(str.strip, str.title))
        l.add_xpath('price', '//*[@class="ad-price txt-xlarge txt-emphasis inline-block"][1]/text()', MapCompose(lambda i: i.replace(',', ''), float), re='[.0-9]+')
        l.add_xpath('description', '//*[@itemprop="description"][1]/text()', MapCompose(str.strip), Join())
        l.add_xpath('address', '//*[@itemprop="address"][1]/text()', MapCompose(str.strip))
        l.add_xpath('image_urls', '//*[@id="image-gallery"]/div/div/div[1]/ul/li[1]/img/@src', MapCompose(lambda i: urljoin(response.url, i)))

        # we can add single values that we calculate with Python

        l.add_value('url', response.url)
        l.add_value('project', self.settings.get('BOT_NAME'))
        l.add_value('spider', self.name)
        l.add_value('server', socket.gethostname())
        l.add_value('date', datetime.datetime.now())

        return l.load_item()


