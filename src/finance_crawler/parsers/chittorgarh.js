const cheerio = require('cheerio');
function parseChittorgarh(html) {
  const $ = cheerio.load(html);
  const ipos = [];
  $('.ipo-listing-table tbody tr').each((i, row) => {
    const cols = $(row).find('td');
    if (cols.length >= 4) {
      ipos.push({
        type: 'ipo',
        name: $(cols[0]).text().trim(),
        date: $(cols[1]).text().trim(),
        status: $(cols[2]).text().trim(),
        link: 'https://www.chittorgarh.com' + $(cols[0]).find('a').attr('href'),
        crawled_at: new Date().toISOString()
      });
    }
  });
  return ipos;
}
module.exports = parseChittorgarh;
