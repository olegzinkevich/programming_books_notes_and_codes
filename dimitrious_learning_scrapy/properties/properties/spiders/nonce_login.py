# -*- coding: utf-8 -*-


import scrapy

import datetime
from urllib.parse import urljoin
import socket
import scrapy

from scrapy.loader.processors import MapCompose, Join
from scrapy.loader import ItemLoader
from scrapy.http import Request
from scrapy.http import FormRequest
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule

from properties.items import PropertiesItem

try:
    from urllib import urlencode
except ImportError:
    from urllib.parse import urlencode


class NonceLoginSpider(CrawlSpider):

    name = "nonce_login"
    allowed_domains = ["localhost"]

    # When you submit this form
    # (to http://localhost:9312/dynamic/nonce-login), the login won't be successful unless you pass not only the correct user/pass, but also the exact nonce value that server gave you when you visited this login page. There is no way for you to guess that value as it typically will be random and single-use. This means that in order to successfully log in, you now need two requests.

    # Start on the welcome page
    def start_requests(self):
        return [
            Request(
                "http://localhost:9312/dynamic/login",
                callback=self.parse_welcome)
        ]

    # Post welcome page's first form with the given user/pass
    def parse_welcome(self, response):
        return FormRequest.from_response(
            response,
            formdata={"user": "user", "pass": "pass"}
        )


    rules = (
        #  As their name implies, LinkExtractors are specialized in extracting links, so by default, they are looking for the a (and area) href attributes. Finally, unless callback is set, a Rule will follow the extracted URLs, which means that it will scan target pages for extra links and follow them.
        Rule(LinkExtractor(restrict_xpaths='//*[contains(@class,"next")]')),

        #  If a callback is set, the Rule won't follow the links from target pages. If you would like it to follow links, you should either return/yield them from your callback method, or set the follow argument of Rule() to true
        Rule(LinkExtractor(restrict_xpaths='//*[@itemprop="url"]'),
             callback='parse_item')
    )

    def parse_item(self, response):
        """ This function parses a property page.

        @url http://web:9312/properties/property_000000.html
        @returns items 1
        @scrapes title price description address image_urls
        @scrapes url project spider server date
        """

        # Create the loader using the response
        l = ItemLoader(item=PropertiesItem(), response=response)

        # Load fields using XPath expressions
        l.add_xpath('title', '//*[@itemprop="name"][1]/text()',
                    MapCompose(str.strip, str.title))
        l.add_xpath('price', './/*[@itemprop="price"][1]/text()',
                    MapCompose(lambda i: i.replace(',', ''), float),
                    re='[,.0-9]+')
        l.add_xpath('description', '//*[@itemprop="description"][1]/text()',
                    MapCompose(str.strip), Join())
        l.add_xpath('address',
                    '//*[@itemtype="http://schema.org/Place"][1]/text()',
                    MapCompose(str.strip))
        l.add_xpath('image_urls', '//*[@itemprop="image"][1]/@src',
                    MapCompose(lambda i: urljoin(response.url, i)))

        # Housekeeping fields
        l.add_value('url', response.url)
        l.add_value('project', self.settings.get('BOT_NAME'))
        l.add_value('spider', self.name)
        l.add_value('server', socket.gethostname())
        l.add_value('date', datetime.datetime.now())

        return l.load_item()