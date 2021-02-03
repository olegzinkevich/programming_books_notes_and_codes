# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy
from scrapy.item import Item, Field

class PropertiesItem(Item):
    # Primary fields
    title = Field()
    price = Field()
    description = Field()
    address = Field()
    image_urls = Field()

    # Calculated fields
    images = Field()
    location = Field()

    # Housekeeping fields. If you have a look at them, you'll understand that they allow me to find out where (server, url), when (date), and how (spider) an item got scraped.
    url = Field()
    project = Field()
    spider = Field()
    server = Field()
    date = Field()
