import React, { useState, useRef, useEffect } from 'react';

// Dynamically import Lightweight Charts to avoid SSR issues
let createChart = null;
if (typeof window !== 'undefined') {
  import('lightweight-charts').then(m => { createChart = m.createChart; });
}

const STATUS_STEPS = [
  { progress: 10, message: 'Fetching latest market data…' },
  { progress: 30, message: 'Running AI prediction models…' },
  { progress: 60, message: 'Synthesizing insights…' },
  { progress: 90, message: 'Finalizing analysis…' },
  { progress: 100, message: 'Analysis complete!' },
];

export default function MutualFundAnalysis({ fundId = 'default-fund' }) {
  const [jobId, setJobId] = useState(null);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState('idle');
  const [message, setMessage] = useState('');
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const pollingRef = useRef(null);

  const startAnalysis = async () => {
    setStatus('pending');
    setProgress(0);
    setMessage('Starting analysis…');
    setResult(null);
    setError(null);
    try {
      const res = await fetch('/api/analyze/analyze-mutual-fund', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fundId }),
      });
      const data = await res.json();
      setJobId(data.jobId);
      pollStatus(data.jobId);
    } catch (err) {
      setError('Failed to start analysis.');
      setStatus('idle');
    }
  };

  const pollStatus = (jobId) => {
    pollingRef.current = setInterval(async () => {
      try {
        const res = await fetch(`/api/analyze/job-status/${jobId}`);
        const data = await res.json();
        setProgress(data.progress);
        setMessage(data.message);
        setStatus(data.status);
        if (data.status === 'completed') {
          setResult(data.result);
          clearInterval(pollingRef.current);
        }
        if (data.status === 'failed') {
          setError(data.error || 'Analysis failed.');
          clearInterval(pollingRef.current);
        }
      } catch (err) {
        setError('Error fetching status.');
        clearInterval(pollingRef.current);
      }
    }, 2500);
  };

  // Clean up polling on unmount
  React.useEffect(() => {
    return () => {
      if (pollingRef.current) clearInterval(pollingRef.current);
    };
  }, []);

  return (
    <div className="max-w-xl mx-auto bg-gray-900 rounded-xl shadow-lg p-8 mt-8 text-white">
      <h2 className="text-2xl font-bold mb-4 text-neon-green drop-shadow-lg">Mutual Fund Premium Analysis</h2>
      {status === 'idle' && (
        <button
          onClick={startAnalysis}
          className="bg-gradient-to-r from-green-400 via-emerald-400 to-neon-green text-gray-900 px-6 py-3 rounded-lg shadow-lg font-semibold text-lg hover:scale-105 transition-transform duration-200"
        >
          Analyze Mutual Fund
        </button>
      )}
      {(status === 'pending' || status === 'running') && (
        <div className="mt-6">
          <div className="w-full bg-gray-800 rounded-full h-6 mb-4 overflow-hidden">
            <div
              className="h-6 bg-gradient-to-r from-neon-green via-green-400 to-emerald-400 animate-pulse"
              style={{ width: `${progress}%`, transition: 'width 0.5s' }}
            ></div>
          </div>
          <div className="text-lg font-medium animate-pulse">{message}</div>
          <div className="mt-2 text-sm text-gray-400">This may take up to 15 seconds. Please wait…</div>
        </div>
      )}
      {status === 'completed' && result && (
        <div className="mt-6">
          <div className="text-xl font-bold text-neon-green mb-2 animate-fade-in">Analysis Complete!</div>
          <div className="bg-gray-800 rounded-lg p-4 shadow-inner">
            <ul className="list-disc list-inside space-y-2">
              {result.insights.map((insight, idx) => (
                <li key={idx} className="text-lg text-emerald-300 animate-fade-in-slow">
                  {insight}
                </li>
              ))}
            </ul>
            <div className="mt-4 text-xs text-gray-400">Generated at: {new Date(result.generatedAt).toLocaleString()}</div>
          </div>
          {/* Render chart if chartData is present */}
          {result.chartData && result.chartData.length > 0 && (
            <TradingViewChart chartData={result.chartData} />
          )}
          <button
            onClick={() => setStatus('idle')}
            className="mt-6 bg-gradient-to-r from-emerald-400 to-neon-green text-gray-900 px-5 py-2 rounded-lg font-semibold shadow hover:scale-105 transition-transform"
          >
            Analyze Another Fund
          </button>
        </div>
      )}
      {error && (
        <div className="mt-4 text-red-400 font-semibold">{error}</div>
      )}
    </div>
  );
}

// TradingView-style chart using Lightweight Charts
function TradingViewChart({ chartData }) {
  const chartRef = useRef(null);
  const chartInstance = useRef(null);
  const seriesRef = useRef(null);

  useEffect(() => {
    if (!createChart || !chartRef.current) return;
    // Clear previous chart
    if (chartInstance.current) {
      chartInstance.current.remove();
      chartInstance.current = null;
    }
    // Create chart
    chartInstance.current = createChart(chartRef.current, {
      width: chartRef.current.offsetWidth || 700,
      height: 350,
      layout: { background: { color: '#18181b' }, textColor: '#fff' },
      grid: { vertLines: { color: '#222' }, horzLines: { color: '#222' } },
      timeScale: { timeVisible: true, secondsVisible: false },
      crosshair: { mode: 1 },
      handleScroll: true,
      handleScale: true,
    });
    seriesRef.current = chartInstance.current.addCandlestickSeries();
    seriesRef.current.setData(chartData.map(bar => ({
      time: bar.time, open: bar.open, high: bar.high, low: bar.low, close: bar.close
    })));
    // Optional: Add volume as histogram
    if (chartData[0]?.volume) {
      const volSeries = chartInstance.current.addHistogramSeries({
        color: '#39ff14', priceFormat: { type: 'volume' },
        priceScaleId: '',
        scaleMargins: { top: 0.8, bottom: 0 },
      });
      volSeries.setData(chartData.map(bar => ({ time: bar.time, value: bar.volume })));
    }
    // Responsive resize
    const handleResize = () => {
      if (chartInstance.current && chartRef.current) {
        chartInstance.current.applyOptions({ width: chartRef.current.offsetWidth });
      }
    };
    window.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('resize', handleResize);
      if (chartInstance.current) chartInstance.current.remove();
    };
  }, [chartData]);

  return (
    <div className="my-8">
      <div ref={chartRef} style={{ width: '100%', minHeight: 350 }} />
      <div className="text-xs text-gray-500 mt-2">
        Interactive chart: scroll, zoom, hover bars for details. Powered by TradingView Lightweight Charts.
      </div>
    </div>
  );
}

// Tailwind custom color: add to tailwind.config.js
// colors: { 'neon-green': '#39ff14', ... }
