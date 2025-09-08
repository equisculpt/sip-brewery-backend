import scrapy
import re

INDIAN_FINANCE_KEYWORDS = [
    "nse", "bse", "sebi", "rbi", "mutual fund", "sip", "india", "indian stock", "sensex", "nifty", "inr", "amfi",
    "hdfc", "icici", "kotak", "sbi", "axis bank", "reliance", "tata", "birla", "l&t", "bajaj", "fintech", "ipo"
]

class IndianFinanceSpider(scrapy.Spider):
    name = "indian_finance"
    start_urls = [
        "https://www.moneycontrol.com/", "https://economictimes.indiatimes.com/", "https://www.reuters.com/markets/",
        "https://www.sebi.gov.in/", "https://www.rbi.org.in/", "https://www.bseindia.com/", "https://www.nseindia.com/"
    ]

    def parse(self, response):
        text = " ".join(response.xpath("//body//text()").getall()).lower()
        if any(k in text for k in INDIAN_FINANCE_KEYWORDS):
            yield {
                "url": response.url,
                "title": response.xpath("//title/text()").get(),
                "content": text,
            }
        for link in response.css("a::attr(href)").getall():
            if link and link.startswith("http"):
                yield response.follow(link, self.parse)
