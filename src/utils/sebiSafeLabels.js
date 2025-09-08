// SEBI-safe label mapping and disclaimer helper

const SEBI_SAFE_LABELS = {
  buy: {
    label: 'Potential Accumulation Zone',
    info: 'For your information: This asset is in a potential accumulation zone based on historical data.'
  },
  sell: {
    label: 'Potential Distribution Zone',
    info: 'Educational note: Reduced momentum observed. You may wish to review your position.'
  },
  hold: {
    label: 'Stable Zone',
    info: 'Informational update: No significant change detected. Maintaining your current allocation aligns with recent data.'
  },
  default: {
    label: 'Stable Zone',
    info: 'Informational update: No significant change detected. Maintaining your current allocation aligns with recent data.'
  }
};

const SEBI_DISCLAIMER = 'This information is provided for educational and informational purposes only and should not be construed as investment advice. Please consult a SEBI-registered advisor for actionable decisions.';

function getSebiSafeLabel(signal) {
  const key = (signal || '').toLowerCase();
  return SEBI_SAFE_LABELS[key] || SEBI_SAFE_LABELS.default;
}

function getSebiDisclaimer() {
  return SEBI_DISCLAIMER;
}

module.exports = {
  getSebiSafeLabel,
  getSebiDisclaimer
};
