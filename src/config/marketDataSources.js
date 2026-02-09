module.exports = {
  sources: [
    {
      name: 'amfi_nav',
      type: 'http',
      enabled: false,
      endpoint: 'https://www.amfiindia.com/spages/NAVAll.txt'
    },
    {
      name: 'bse_star_mf',
      type: 'http',
      enabled: false,
      endpoint: 'https://api.bseindia.com/BseIndiaAPI/api/MutualFund'
    }
  ]
};
