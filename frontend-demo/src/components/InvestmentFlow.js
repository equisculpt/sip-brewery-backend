import React, { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { portfolioAPI, bseStarMFAPI } from '../services/api';
import { 
  Search, 
  Filter, 
  Star, 
  TrendingUp, 
  TrendingDown, 
  ArrowRight,
  CheckCircle,
  Clock,
  AlertCircle,
  DollarSign,
  Calendar,
  Target
} from 'lucide-react';
import toast from 'react-hot-toast';

const InvestmentFlow = () => {
  const { user } = useAuth();
  const [step, setStep] = useState(1);
  const [funds, setFunds] = useState([]);
  const [selectedFund, setSelectedFund] = useState(null);
  const [investmentAmount, setInvestmentAmount] = useState(5000);
  const [investmentType, setInvestmentType] = useState('lumpsum');
  const [sipDetails, setSipDetails] = useState({
    frequency: 'monthly',
    duration: 12
  });
  const [orderData, setOrderData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [filters, setFilters] = useState({
    category: '',
    risk: '',
    fundHouse: ''
  });

  useEffect(() => {
    loadFunds();
  }, [filters]);

  const loadFunds = async () => {
    try {
      setLoading(true);
      const response = await bseStarMFAPI.getSchemes({
        limit: 20,
        ...filters
      });
      setFunds(response.schemes || []);
    } catch (error) {
      toast.error('Failed to load funds');
    } finally {
      setLoading(false);
    }
  };

  const handleFundSelect = (fund) => {
    setSelectedFund(fund);
    setStep(2);
  };

  const handleInvestmentSubmit = async () => {
    if (!selectedFund) {
      toast.error('Please select a fund first');
      return;
    }

    if (investmentAmount < selectedFund.minInvestment) {
      toast.error(`Minimum investment amount is ${selectedFund.minInvestment}`);
      return;
    }

    try {
      setLoading(true);
      
      // Create client first (if not exists)
      const clientData = {
        clientData: {
          firstName: user.name.split(' ')[0],
          lastName: user.name.split(' ').slice(1).join(' '),
          dateOfBirth: '1990-01-01',
          gender: 'MALE',
          panNumber: 'ABCDE1234F',
          aadhaarNumber: '123456789012',
          email: user.email,
          mobile: '9876543210',
          address: {
            line1: '123 Main Street',
            city: 'Mumbai',
            state: 'Maharashtra',
            pincode: '400001'
          },
          bankDetails: {
            accountNumber: '1234567890',
            ifscCode: 'SBIN0001234',
            accountHolderName: user.name
          }
        }
      };

      const clientResponse = await bseStarMFAPI.createClient(clientData);
      const clientId = clientResponse.clientId;

      // Place investment order
      const orderData = {
        orderData: {
          clientId,
          schemeCode: selectedFund.schemeCode,
          amount: investmentAmount,
          paymentMode: 'ONLINE',
          isSmartSIP: investmentType === 'sip'
        }
      };

      const orderResponse = await bseStarMFAPI.placeOrder(orderData);
      setOrderData(orderResponse);
      setStep(3);
      
      toast.success('Investment order placed successfully!');
    } catch (error) {
      toast.error('Failed to place investment order');
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

  const getRiskColor = (risk) => {
    switch (risk.toLowerCase()) {
      case 'low': return 'text-green-600 bg-green-100';
      case 'moderate': return 'text-yellow-600 bg-yellow-100';
      case 'high': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getCategoryColor = (category) => {
    switch (category.toLowerCase()) {
      case 'equity': return 'text-blue-600 bg-blue-100';
      case 'debt': return 'text-green-600 bg-green-100';
      case 'gold': return 'text-yellow-600 bg-yellow-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">Invest in Mutual Funds</h1>
          <p className="text-gray-600 mt-2">Choose from our curated selection of top-performing funds</p>
        </div>

        {/* Progress Steps */}
        <div className="mb-8">
          <div className="flex items-center justify-center">
            <div className="flex items-center">
              <div className={`flex items-center justify-center w-10 h-10 rounded-full ${
                step >= 1 ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-600'
              }`}>
                1
              </div>
              <div className={`ml-4 ${step >= 1 ? 'text-blue-600' : 'text-gray-400'}`}>
                Select Fund
              </div>
            </div>
            
            <div className={`w-16 h-0.5 mx-4 ${step >= 2 ? 'bg-blue-600' : 'bg-gray-200'}`}></div>
            
            <div className="flex items-center">
              <div className={`flex items-center justify-center w-10 h-10 rounded-full ${
                step >= 2 ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-600'
              }`}>
                2
              </div>
              <div className={`ml-4 ${step >= 2 ? 'text-blue-600' : 'text-gray-400'}`}>
                Investment Details
              </div>
            </div>
            
            <div className={`w-16 h-0.5 mx-4 ${step >= 3 ? 'bg-blue-600' : 'bg-gray-200'}`}></div>
            
            <div className="flex items-center">
              <div className={`flex items-center justify-center w-10 h-10 rounded-full ${
                step >= 3 ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-600'
              }`}>
                3
              </div>
              <div className={`ml-4 ${step >= 3 ? 'text-blue-600' : 'text-gray-400'}`}>
                Order Confirmation
              </div>
            </div>
          </div>
        </div>

        {/* Step 1: Fund Selection */}
        {step === 1 && (
          <div className="bg-white rounded-lg shadow">
            <div className="p-6 border-b border-gray-200">
              <h2 className="text-xl font-semibold text-gray-900">Select a Fund</h2>
              
              {/* Filters */}
              <div className="mt-4 flex flex-wrap gap-4">
                <select
                  value={filters.category}
                  onChange={(e) => setFilters({...filters, category: e.target.value})}
                  className="px-3 py-2 border border-gray-300 rounded-md text-sm"
                >
                  <option value="">All Categories</option>
                  <option value="equity">Equity</option>
                  <option value="debt">Debt</option>
                  <option value="gold">Gold</option>
                </select>
                
                <select
                  value={filters.risk}
                  onChange={(e) => setFilters({...filters, risk: e.target.value})}
                  className="px-3 py-2 border border-gray-300 rounded-md text-sm"
                >
                  <option value="">All Risk Levels</option>
                  <option value="low">Low Risk</option>
                  <option value="moderate">Moderate Risk</option>
                  <option value="high">High Risk</option>
                </select>
                
                <select
                  value={filters.fundHouse}
                  onChange={(e) => setFilters({...filters, fundHouse: e.target.value})}
                  className="px-3 py-2 border border-gray-300 rounded-md text-sm"
                >
                  <option value="">All Fund Houses</option>
                  <option value="HDFC">HDFC Mutual Fund</option>
                  <option value="ICICI">ICICI Prudential</option>
                  <option value="SBI">SBI Mutual Fund</option>
                  <option value="Axis">Axis Mutual Fund</option>
                </select>
              </div>
            </div>
            
            <div className="p-6">
              {loading ? (
                <div className="flex items-center justify-center py-12">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {funds.map((fund) => (
                    <div
                      key={fund.schemeCode}
                      className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow cursor-pointer"
                      onClick={() => handleFundSelect(fund)}
                    >
                      <div className="flex items-start justify-between mb-3">
                        <div className="flex-1">
                          <h3 className="font-semibold text-gray-900 text-sm mb-1">{fund.schemeName}</h3>
                          <p className="text-gray-600 text-xs">{fund.fundHouse}</p>
                        </div>
                        <div className="flex items-center gap-1">
                          <Star className="w-4 h-4 text-yellow-400 fill-current" />
                          <span className="text-sm font-medium">{fund.rating}</span>
                        </div>
                      </div>
                      
                      <div className="space-y-2 mb-4">
                        <div className="flex items-center justify-between">
                          <span className="text-xs text-gray-600">NAV:</span>
                          <span className="text-sm font-medium">₹{fund.nav}</span>
                        </div>
                        
                        <div className="flex items-center justify-between">
                          <span className="text-xs text-gray-600">Min Investment:</span>
                          <span className="text-sm font-medium">{formatCurrency(fund.minInvestment)}</span>
                        </div>
                      </div>
                      
                      <div className="flex flex-wrap gap-2">
                        <span className={`px-2 py-1 text-xs rounded-full ${getCategoryColor(fund.category)}`}>
                          {fund.category}
                        </span>
                        <span className={`px-2 py-1 text-xs rounded-full ${getRiskColor(fund.riskLevel)}`}>
                          {fund.riskLevel}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Step 2: Investment Details */}
        {step === 2 && selectedFund && (
          <div className="bg-white rounded-lg shadow">
            <div className="p-6 border-b border-gray-200">
              <h2 className="text-xl font-semibold text-gray-900">Investment Details</h2>
            </div>
            
            <div className="p-6">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* Selected Fund Info */}
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Selected Fund</h3>
                  <div className="border border-gray-200 rounded-lg p-4">
                    <h4 className="font-semibold text-gray-900 mb-2">{selectedFund.schemeName}</h4>
                    <p className="text-gray-600 text-sm mb-3">{selectedFund.fundHouse}</p>
                    
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span className="text-sm text-gray-600">NAV:</span>
                        <span className="text-sm font-medium">₹{selectedFund.nav}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm text-gray-600">Category:</span>
                        <span className={`px-2 py-1 text-xs rounded-full ${getCategoryColor(selectedFund.category)}`}>
                          {selectedFund.category}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm text-gray-600">Risk Level:</span>
                        <span className={`px-2 py-1 text-xs rounded-full ${getRiskColor(selectedFund.riskLevel)}`}>
                          {selectedFund.riskLevel}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm text-gray-600">Rating:</span>
                        <div className="flex items-center gap-1">
                          <Star className="w-4 h-4 text-yellow-400 fill-current" />
                          <span className="text-sm font-medium">{selectedFund.rating}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Investment Form */}
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Investment Details</h3>
                  
                  <div className="space-y-4">
                    {/* Investment Type */}
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">Investment Type</label>
                      <div className="flex gap-4">
                        <label className="flex items-center">
                          <input
                            type="radio"
                            value="lumpsum"
                            checked={investmentType === 'lumpsum'}
                            onChange={(e) => setInvestmentType(e.target.value)}
                            className="mr-2"
                          />
                          <span className="text-sm">Lumpsum</span>
                        </label>
                        <label className="flex items-center">
                          <input
                            type="radio"
                            value="sip"
                            checked={investmentType === 'sip'}
                            onChange={(e) => setInvestmentType(e.target.value)}
                            className="mr-2"
                          />
                          <span className="text-sm">SIP</span>
                        </label>
                      </div>
                    </div>

                    {/* Investment Amount */}
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Investment Amount
                      </label>
                      <div className="relative">
                        <DollarSign className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                        <input
                          type="number"
                          value={investmentAmount}
                          onChange={(e) => setInvestmentAmount(Number(e.target.value))}
                          min={selectedFund.minInvestment}
                          className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                          placeholder="Enter amount"
                        />
                      </div>
                      <p className="text-xs text-gray-500 mt-1">
                        Minimum investment: {formatCurrency(selectedFund.minInvestment)}
                      </p>
                    </div>

                    {/* SIP Details (if SIP selected) */}
                    {investmentType === 'sip' && (
                      <div className="space-y-4 p-4 bg-gray-50 rounded-lg">
                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-2">SIP Frequency</label>
                          <select
                            value={sipDetails.frequency}
                            onChange={(e) => setSipDetails({...sipDetails, frequency: e.target.value})}
                            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                          >
                            <option value="monthly">Monthly</option>
                            <option value="weekly">Weekly</option>
                            <option value="quarterly">Quarterly</option>
                          </select>
                        </div>
                        
                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-2">Duration (months)</label>
                          <input
                            type="number"
                            value={sipDetails.duration}
                            onChange={(e) => setSipDetails({...sipDetails, duration: Number(e.target.value)})}
                            min="1"
                            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                          />
                        </div>
                      </div>
                    )}

                    {/* Estimated Units */}
                    <div className="p-4 bg-blue-50 rounded-lg">
                      <div className="flex justify-between items-center">
                        <span className="text-sm font-medium text-gray-700">Estimated Units:</span>
                        <span className="text-lg font-semibold text-blue-600">
                          {(investmentAmount / selectedFund.nav).toFixed(2)}
                        </span>
                      </div>
                    </div>

                    {/* Action Buttons */}
                    <div className="flex gap-4 pt-4">
                      <button
                        onClick={() => setStep(1)}
                        className="flex-1 px-4 py-2 border border-gray-300 text-gray-700 rounded-md hover:bg-gray-50 transition-colors"
                      >
                        Back
                      </button>
                      <button
                        onClick={handleInvestmentSubmit}
                        disabled={loading}
                        className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors disabled:opacity-50"
                      >
                        {loading ? 'Processing...' : 'Invest Now'}
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Step 3: Order Confirmation */}
        {step === 3 && orderData && (
          <div className="bg-white rounded-lg shadow">
            <div className="p-6 border-b border-gray-200">
              <h2 className="text-xl font-semibold text-gray-900">Order Confirmation</h2>
            </div>
            
            <div className="p-6">
              <div className="text-center mb-6">
                <CheckCircle className="w-16 h-16 text-green-600 mx-auto mb-4" />
                <h3 className="text-xl font-semibold text-gray-900 mb-2">Order Placed Successfully!</h3>
                <p className="text-gray-600">Your investment order has been submitted and is being processed.</p>
              </div>
              
              <div className="max-w-md mx-auto">
                <div className="bg-gray-50 rounded-lg p-6 space-y-4">
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Order ID:</span>
                    <span className="text-sm font-medium">{orderData.orderId}</span>
                  </div>
                  
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Fund:</span>
                    <span className="text-sm font-medium">{selectedFund.schemeName}</span>
                  </div>
                  
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Amount:</span>
                    <span className="text-sm font-medium">{formatCurrency(investmentAmount)}</span>
                  </div>
                  
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Type:</span>
                    <span className="text-sm font-medium capitalize">{investmentType}</span>
                  </div>
                  
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Status:</span>
                    <span className="px-2 py-1 text-xs bg-yellow-100 text-yellow-800 rounded-full">
                      {orderData.status}
                    </span>
                  </div>
                  
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Order Date:</span>
                    <span className="text-sm font-medium">
                      {new Date(orderData.orderDate).toLocaleDateString()}
                    </span>
                  </div>
                </div>
                
                <div className="mt-6 space-y-3">
                  <button
                    onClick={() => window.location.href = '/dashboard'}
                    className="w-full px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
                  >
                    Go to Dashboard
                  </button>
                  
                  <button
                    onClick={() => setStep(1)}
                    className="w-full px-4 py-2 border border-gray-300 text-gray-700 rounded-md hover:bg-gray-50 transition-colors"
                  >
                    Invest in Another Fund
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default InvestmentFlow; 