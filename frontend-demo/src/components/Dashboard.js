import React, { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { portfolioAPI, aiAPI, analyticsAPI } from '../services/api';
import { 
  TrendingUp, 
  TrendingDown, 
  DollarSign, 
  PieChart, 
  Target, 
  Award,
  Calendar,
  ArrowUpRight,
  ArrowDownRight,
  Eye,
  Plus,
  Download
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart as RechartsPieChart, Pie, Cell } from 'recharts';
import toast from 'react-hot-toast';

const Dashboard = () => {
  const { user } = useAuth();
  const [portfolioData, setPortfolioData] = useState(null);
  const [aiInsights, setAiInsights] = useState(null);
  const [dashboardData, setDashboardData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    try {
      setLoading(true);
      
      // Load all dashboard data in parallel
      const [portfolioRes, insightsRes, analyticsRes] = await Promise.allSettled([
        portfolioAPI.getOverview(),
        aiAPI.getPortfolioInsights(),
        analyticsAPI.getDashboardData()
      ]);

      if (portfolioRes.status === 'fulfilled') {
        setPortfolioData(portfolioRes.value.data);
      }

      if (insightsRes.status === 'fulfilled') {
        setAiInsights(insightsRes.value.data);
      }

      if (analyticsRes.status === 'fulfilled') {
        setDashboardData(analyticsRes.value.data);
      }

    } catch (error) {
      toast.error('Failed to load dashboard data');
    } finally {
      setLoading(false);
    }
  };

  const formatCurrency = (amount) => {
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(amount);
  };

  const formatPercentage = (value) => {
    return `${value > 0 ? '+' : ''}${value.toFixed(2)}%`;
  };

  const getPerformanceColor = (value) => {
    return value >= 0 ? 'text-green-600' : 'text-red-600';
  };

  const getPerformanceIcon = (value) => {
    return value >= 0 ? <ArrowUpRight className="w-4 h-4" /> : <ArrowDownRight className="w-4 h-4" />;
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">Welcome back, {user?.name}!</h1>
          <p className="text-gray-600 mt-2">Here's what's happening with your investments today</p>
        </div>

        {/* Portfolio Overview Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Total Invested</p>
                <p className="text-2xl font-bold text-gray-900">
                  {portfolioData ? formatCurrency(portfolioData.totalInvested) : '₹0'}
                </p>
              </div>
              <div className="p-3 bg-blue-100 rounded-full">
                <DollarSign className="w-6 h-6 text-blue-600" />
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Current Value</p>
                <p className="text-2xl font-bold text-gray-900">
                  {portfolioData ? formatCurrency(portfolioData.totalCurrentValue) : '₹0'}
                </p>
              </div>
              <div className="p-3 bg-green-100 rounded-full">
                <TrendingUp className="w-6 h-6 text-green-600" />
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Absolute Return</p>
                <p className={`text-2xl font-bold ${getPerformanceColor(portfolioData?.absoluteReturnPercent || 0)}`}>
                  {portfolioData ? formatCurrency(portfolioData.absoluteReturn) : '₹0'}
                </p>
                <p className={`text-sm flex items-center gap-1 ${getPerformanceColor(portfolioData?.absoluteReturnPercent || 0)}`}>
                  {getPerformanceIcon(portfolioData?.absoluteReturnPercent || 0)}
                  {portfolioData ? formatPercentage(portfolioData.absoluteReturnPercent) : '0%'}
                </p>
              </div>
              <div className="p-3 bg-purple-100 rounded-full">
                <Target className="w-6 h-6 text-purple-600" />
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">XIRR (1Y)</p>
                <p className={`text-2xl font-bold ${getPerformanceColor(dashboardData?.performance?.xirr1Y || 0)}`}>
                  {dashboardData ? `${(dashboardData.performance.xirr1Y * 100).toFixed(2)}%` : '0%'}
                </p>
              </div>
              <div className="p-3 bg-orange-100 rounded-full">
                <Award className="w-6 h-6 text-orange-600" />
              </div>
            </div>
          </div>
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Portfolio Performance Chart */}
          <div className="lg:col-span-2 bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-semibold text-gray-900">Portfolio Performance</h2>
              <div className="flex gap-2">
                <button className="px-3 py-1 text-sm bg-blue-100 text-blue-600 rounded-md">1M</button>
                <button className="px-3 py-1 text-sm bg-gray-100 text-gray-600 rounded-md">3M</button>
                <button className="px-3 py-1 text-sm bg-gray-100 text-gray-600 rounded-md">1Y</button>
              </div>
            </div>
            
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={[
                  { date: 'Jan', value: 100000 },
                  { date: 'Feb', value: 105000 },
                  { date: 'Mar', value: 102000 },
                  { date: 'Apr', value: 108000 },
                  { date: 'May', value: 112000 },
                  { date: 'Jun', value: 115000 },
                ]}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis />
                  <Tooltip formatter={(value) => formatCurrency(value)} />
                  <Line type="monotone" dataKey="value" stroke="#3B82F6" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* AI Insights */}
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-semibold text-gray-900">AI Insights</h2>
              <Eye className="w-5 h-5 text-gray-400" />
            </div>
            
            {aiInsights?.insights?.slice(0, 3).map((insight, index) => (
              <div key={index} className="mb-4 p-4 bg-gray-50 rounded-lg">
                <div className="flex items-start gap-3">
                  <div className={`p-2 rounded-full ${
                    insight.priority === 'high' ? 'bg-red-100' : 
                    insight.priority === 'medium' ? 'bg-yellow-100' : 'bg-green-100'
                  }`}>
                    <Target className={`w-4 h-4 ${
                      insight.priority === 'high' ? 'text-red-600' : 
                      insight.priority === 'medium' ? 'text-yellow-600' : 'text-green-600'
                    }`} />
                  </div>
                  <div className="flex-1">
                    <h3 className="font-medium text-gray-900 text-sm">{insight.title}</h3>
                    <p className="text-gray-600 text-xs mt-1">{insight.description}</p>
                    <div className="flex items-center gap-2 mt-2">
                      <span className="text-xs text-gray-500">Confidence:</span>
                      <span className="text-xs font-medium text-blue-600">
                        {(insight.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Holdings and Quick Actions */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mt-8">
          {/* Portfolio Holdings */}
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-semibold text-gray-900">Portfolio Holdings</h2>
              <PieChart className="w-5 h-5 text-gray-400" />
            </div>
            
            {portfolioData?.holdings?.slice(0, 5).map((holding, index) => (
              <div key={index} className="flex items-center justify-between py-3 border-b border-gray-100 last:border-b-0">
                <div className="flex-1">
                  <h3 className="font-medium text-gray-900 text-sm">{holding.fundName}</h3>
                  <p className="text-gray-600 text-xs">{holding.fundCode}</p>
                </div>
                <div className="text-right">
                  <p className="font-medium text-gray-900 text-sm">
                    {formatCurrency(holding.currentValue)}
                  </p>
                  <p className={`text-xs ${getPerformanceColor(holding.returnPercent)}`}>
                    {formatPercentage(holding.returnPercent)}
                  </p>
                </div>
              </div>
            ))}
            
            {portfolioData?.holdings?.length > 5 && (
              <button className="w-full mt-4 text-sm text-blue-600 hover:text-blue-700">
                View all {portfolioData.holdings.length} holdings
              </button>
            )}
          </div>

          {/* Quick Actions */}
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-6">Quick Actions</h2>
            
            <div className="space-y-4">
              <button className="w-full flex items-center justify-between p-4 bg-blue-50 hover:bg-blue-100 rounded-lg transition-colors">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-blue-100 rounded-full">
                    <Plus className="w-5 h-5 text-blue-600" />
                  </div>
                  <div className="text-left">
                    <h3 className="font-medium text-gray-900">Invest Now</h3>
                    <p className="text-sm text-gray-600">Add to your portfolio</p>
                  </div>
                </div>
                <ArrowUpRight className="w-5 h-5 text-gray-400" />
              </button>

              <button className="w-full flex items-center justify-between p-4 bg-green-50 hover:bg-green-100 rounded-lg transition-colors">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-green-100 rounded-full">
                    <Calendar className="w-5 h-5 text-green-600" />
                  </div>
                  <div className="text-left">
                    <h3 className="font-medium text-gray-900">Setup SIP</h3>
                    <p className="text-sm text-gray-600">Automate your investments</p>
                  </div>
                </div>
                <ArrowUpRight className="w-5 h-5 text-gray-400" />
              </button>

              <button className="w-full flex items-center justify-between p-4 bg-purple-50 hover:bg-purple-100 rounded-lg transition-colors">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-purple-100 rounded-full">
                    <Download className="w-5 h-5 text-purple-600" />
                  </div>
                  <div className="text-left">
                    <h3 className="font-medium text-gray-900">Download Statement</h3>
                    <p className="text-sm text-gray-600">Get your portfolio report</p>
                  </div>
                </div>
                <ArrowUpRight className="w-5 h-5 text-gray-400" />
              </button>
            </div>
          </div>
        </div>

        {/* Recent Transactions */}
        <div className="mt-8 bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-semibold text-gray-900">Recent Transactions</h2>
            <button className="text-sm text-blue-600 hover:text-blue-700">View All</button>
          </div>
          
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-200">
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-600">Date</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-600">Type</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-600">Fund</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-600">Amount</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-600">Status</th>
                </tr>
              </thead>
              <tbody>
                {dashboardData?.recentTransactions?.slice(0, 5).map((transaction, index) => (
                  <tr key={index} className="border-b border-gray-100">
                    <td className="py-3 px-4 text-sm text-gray-900">
                      {new Date(transaction.date).toLocaleDateString()}
                    </td>
                    <td className="py-3 px-4">
                      <span className={`px-2 py-1 text-xs rounded-full ${
                        transaction.type === 'PURCHASE' ? 'bg-green-100 text-green-800' :
                        transaction.type === 'REDEMPTION' ? 'bg-red-100 text-red-800' :
                        'bg-blue-100 text-blue-800'
                      }`}>
                        {transaction.type}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-sm text-gray-900">{transaction.fundName}</td>
                    <td className="py-3 px-4 text-sm font-medium text-gray-900">
                      {formatCurrency(transaction.amount)}
                    </td>
                    <td className="py-3 px-4">
                      <span className={`px-2 py-1 text-xs rounded-full ${
                        transaction.status === 'COMPLETED' ? 'bg-green-100 text-green-800' :
                        transaction.status === 'PENDING' ? 'bg-yellow-100 text-yellow-800' :
                        'bg-red-100 text-red-800'
                      }`}>
                        {transaction.status}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard; 