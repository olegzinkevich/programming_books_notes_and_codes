#  Let's assume that we have an application with several spiders, which provide the crawl date in the usual Python format. Our databases require it in string format in order to index it. We don't want to edit our spiders because there are many of them. How can we do it? A very simple pipeline can postprocess our items and perform the conversion we need.

from datetime import datetime

class TidyUp(object):
    def process_item(self, item, spider):
        item['date'] = map(datetime.isoformat, item['date'])
        return item

# We now have to edit our project's settings.py file and set ITEM_PIPELINES to:

# ITEM_PIPELINES = {'properties.pipelines.tidyup.TidyUp': 100 }
# The number 100 on preceding dict defines the order in which pipelines are going to
# be connected. If another pipeline has a smaller number, it will process Items prior to
# this pipeline.