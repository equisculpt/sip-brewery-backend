import React from 'react';
import MutualFundAnalysis from './components/MutualFundAnalysis';

export default function MutualFundAnalysisPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-950 via-gray-900 to-gray-800 flex flex-col items-center justify-center">
      <h1 className="text-4xl font-extrabold text-neon-green mb-4 drop-shadow-lg text-center">
        Institutional Mutual Fund Analysis
      </h1>
      <p className="text-lg text-gray-300 mb-8 text-center max-w-2xl">
        Experience institutional-grade, AI-powered mutual fund analysis. Our system uses real-time data and advanced AI models to deliver insights that rival top fund managers.
      </p>
      <MutualFundAnalysis />
    </div>
  );
}

// Usage in Next.js (pages directory):
// 1. Copy this file to pages/mutual-fund-analysis.js
// 2. Or, in app router, use as app/mutual-fund-analysis/page.js
// 3. Ensure Tailwind CSS and neon-green color are configured
