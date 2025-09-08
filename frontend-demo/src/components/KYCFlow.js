import React, { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { digioAPI } from '../services/api';
import { 
  User, 
  FileText, 
  Camera, 
  CheckCircle, 
  Clock, 
  AlertCircle,
  ArrowRight,
  Download,
  Eye,
  Shield,
  CreditCard
} from 'lucide-react';
import toast from 'react-hot-toast';

const KYCFlow = () => {
  const { user } = useAuth();
  const [step, setStep] = useState(1);
  const [kycData, setKycData] = useState({
    name: user?.name || '',
    dateOfBirth: '',
    gender: '',
    panNumber: '',
    aadhaarNumber: '',
    mobile: user?.phone || '',
    email: user?.email || '',
    address: {
      line1: '',
      city: '',
      state: '',
      pincode: ''
    }
  });
  const [kycId, setKycId] = useState(null);
  const [kycStatus, setKycStatus] = useState(null);
  const [mandateData, setMandateData] = useState({
    accountNumber: '',
    ifscCode: '',
    accountHolderName: '',
    amount: 10000,
    frequency: 'monthly'
  });
  const [mandateId, setMandateId] = useState(null);
  const [mandateStatus, setMandateStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [panVerification, setPanVerification] = useState(null);

  const handleKYCSubmit = async () => {
    // Validate required fields
    const requiredFields = ['name', 'dateOfBirth', 'gender', 'panNumber', 'aadhaarNumber', 'mobile', 'email'];
    const missingFields = requiredFields.filter(field => !kycData[field]);
    
    if (missingFields.length > 0) {
      toast.error(`Please fill in all required fields: ${missingFields.join(', ')}`);
      return;
    }

    if (!kycData.address.line1 || !kycData.address.city || !kycData.address.state || !kycData.address.pincode) {
      toast.error('Please fill in complete address details');
      return;
    }

    try {
      setLoading(true);
      
      const kycRequestData = {
        kycData: {
          customerDetails: kycData,
          kycType: 'AADHAAR_BASED'
        }
      };

      const response = await digioAPI.initiateKYC(kycRequestData);
      setKycId(response.kycId);
      setStep(2);
      
      toast.success('KYC initiated successfully! Please complete the verification process.');
    } catch (error) {
      toast.error('Failed to initiate KYC. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const checkKYCStatus = async () => {
    if (!kycId) return;

    try {
      const response = await digioAPI.getKYCStatus(kycId);
      setKycStatus(response);
      
      if (response.status === 'COMPLETED') {
        setStep(3);
        toast.success('KYC verification completed successfully!');
      } else if (response.status === 'FAILED') {
        toast.error('KYC verification failed. Please try again.');
      }
    } catch (error) {
      toast.error('Failed to check KYC status');
    }
  };

  const handleMandateSubmit = async () => {
    // Validate mandate data
    if (!mandateData.accountNumber || !mandateData.ifscCode || !mandateData.accountHolderName) {
      toast.error('Please fill in all bank account details');
      return;
    }

    try {
      setLoading(true);
      
      const mandateRequestData = {
        mandateData: {
          customerDetails: {
            name: kycData.name,
            mobile: kycData.mobile,
            email: kycData.email,
            panNumber: kycData.panNumber
          },
          bankDetails: {
            accountNumber: mandateData.accountNumber,
            ifscCode: mandateData.ifscCode,
            accountHolderName: mandateData.accountHolderName
          },
          mandateDetails: {
            amount: mandateData.amount,
            frequency: mandateData.frequency,
            startDate: '2024-02-01',
            endDate: '2025-02-01',
            purpose: 'MUTUAL_FUND_INVESTMENT'
          }
        }
      };

      const response = await digioAPI.setupEMandate(mandateRequestData);
      setMandateId(response.mandateId);
      setStep(4);
      
      toast.success('eMandate setup initiated! Please complete the mandate process.');
    } catch (error) {
      toast.error('Failed to setup eMandate. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const checkMandateStatus = async () => {
    if (!mandateId) return;

    try {
      const response = await digioAPI.getEMandateStatus(mandateId);
      setMandateStatus(response);
      
      if (response.status === 'ACTIVE') {
        setStep(5);
        toast.success('eMandate activated successfully!');
      } else if (response.status === 'REJECTED') {
        toast.error('eMandate was rejected. Please try again.');
      }
    } catch (error) {
      toast.error('Failed to check mandate status');
    }
  };

  const verifyPAN = async () => {
    if (!kycData.panNumber) {
      toast.error('Please enter PAN number first');
      return;
    }

    try {
      setLoading(true);
      const response = await digioAPI.verifyPAN(kycData.panNumber);
      setPanVerification(response);
      
      if (response.status === 'VALID') {
        toast.success('PAN verification successful!');
      } else {
        toast.error('Invalid PAN number. Please check and try again.');
      }
    } catch (error) {
      toast.error('Failed to verify PAN');
    } finally {
      setLoading(false);
    }
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString('en-IN');
  };

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">Complete Your KYC</h1>
          <p className="text-gray-600 mt-2">Verify your identity and setup eMandate for seamless investments</p>
        </div>

        {/* Progress Steps */}
        <div className="mb-8">
          <div className="flex items-center justify-center">
            {[1, 2, 3, 4, 5].map((stepNumber) => (
              <React.Fragment key={stepNumber}>
                <div className="flex items-center">
                  <div className={`flex items-center justify-center w-10 h-10 rounded-full ${
                    step >= stepNumber ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-600'
                  }`}>
                    {stepNumber}
                  </div>
                  <div className={`ml-4 ${step >= stepNumber ? 'text-blue-600' : 'text-gray-400'}`}>
                    {stepNumber === 1 && 'Personal Details'}
                    {stepNumber === 2 && 'KYC Verification'}
                    {stepNumber === 3 && 'KYC Complete'}
                    {stepNumber === 4 && 'eMandate Setup'}
                    {stepNumber === 5 && 'Complete'}
                  </div>
                </div>
                {stepNumber < 5 && (
                  <div className={`w-16 h-0.5 mx-4 ${step >= stepNumber + 1 ? 'bg-blue-600' : 'bg-gray-200'}`}></div>
                )}
              </React.Fragment>
            ))}
          </div>
        </div>

        {/* Step 1: Personal Details */}
        {step === 1 && (
          <div className="bg-white rounded-lg shadow">
            <div className="p-6 border-b border-gray-200">
              <h2 className="text-xl font-semibold text-gray-900">Personal Details</h2>
              <p className="text-gray-600 mt-1">Please provide your personal information for KYC verification</p>
            </div>
            
            <div className="p-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Basic Information */}
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">Full Name *</label>
                    <input
                      type="text"
                      value={kycData.name}
                      onChange={(e) => setKycData({...kycData, name: e.target.value})}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      placeholder="Enter your full name"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">Date of Birth *</label>
                    <input
                      type="date"
                      value={kycData.dateOfBirth}
                      onChange={(e) => setKycData({...kycData, dateOfBirth: e.target.value})}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">Gender *</label>
                    <select
                      value={kycData.gender}
                      onChange={(e) => setKycData({...kycData, gender: e.target.value})}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    >
                      <option value="">Select Gender</option>
                      <option value="MALE">Male</option>
                      <option value="FEMALE">Female</option>
                      <option value="OTHER">Other</option>
                    </select>
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">Mobile Number *</label>
                    <input
                      type="tel"
                      value={kycData.mobile}
                      onChange={(e) => setKycData({...kycData, mobile: e.target.value})}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      placeholder="Enter mobile number"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">Email Address *</label>
                    <input
                      type="email"
                      value={kycData.email}
                      onChange={(e) => setKycData({...kycData, email: e.target.value})}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      placeholder="Enter email address"
                    />
                  </div>
                </div>

                {/* Identity Documents */}
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">PAN Number *</label>
                    <div className="flex gap-2">
                      <input
                        type="text"
                        value={kycData.panNumber}
                        onChange={(e) => setKycData({...kycData, panNumber: e.target.value.toUpperCase()})}
                        className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        placeholder="ABCDE1234F"
                        maxLength="10"
                      />
                      <button
                        onClick={verifyPAN}
                        disabled={loading || !kycData.panNumber}
                        className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors disabled:opacity-50"
                      >
                        Verify
                      </button>
                    </div>
                    {panVerification && (
                      <div className={`mt-2 p-2 rounded-md text-sm ${
                        panVerification.status === 'VALID' 
                          ? 'bg-green-100 text-green-800' 
                          : 'bg-red-100 text-red-800'
                      }`}>
                        {panVerification.status === 'VALID' ? '✓ PAN verified' : '✗ Invalid PAN'}
                      </div>
                    )}
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">Aadhaar Number *</label>
                    <input
                      type="text"
                      value={kycData.aadhaarNumber}
                      onChange={(e) => setKycData({...kycData, aadhaarNumber: e.target.value})}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      placeholder="123456789012"
                      maxLength="12"
                    />
                  </div>
                  
                  {/* Address */}
                  <div className="space-y-3">
                    <h3 className="text-sm font-medium text-gray-700">Address Details *</h3>
                    
                    <input
                      type="text"
                      value={kycData.address.line1}
                      onChange={(e) => setKycData({
                        ...kycData, 
                        address: {...kycData.address, line1: e.target.value}
                      })}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      placeholder="Address Line 1"
                    />
                    
                    <div className="grid grid-cols-2 gap-3">
                      <input
                        type="text"
                        value={kycData.address.city}
                        onChange={(e) => setKycData({
                          ...kycData, 
                          address: {...kycData.address, city: e.target.value}
                        })}
                        className="px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        placeholder="City"
                      />
                      
                      <input
                        type="text"
                        value={kycData.address.state}
                        onChange={(e) => setKycData({
                          ...kycData, 
                          address: {...kycData.address, state: e.target.value}
                        })}
                        className="px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        placeholder="State"
                      />
                    </div>
                    
                    <input
                      type="text"
                      value={kycData.address.pincode}
                      onChange={(e) => setKycData({
                        ...kycData, 
                        address: {...kycData.address, pincode: e.target.value}
                      })}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      placeholder="PIN Code"
                      maxLength="6"
                    />
                  </div>
                </div>
              </div>
              
              <div className="mt-8 flex justify-end">
                <button
                  onClick={handleKYCSubmit}
                  disabled={loading}
                  className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors disabled:opacity-50"
                >
                  {loading ? 'Processing...' : 'Continue to KYC Verification'}
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Step 2: KYC Verification */}
        {step === 2 && (
          <div className="bg-white rounded-lg shadow">
            <div className="p-6 border-b border-gray-200">
              <h2 className="text-xl font-semibold text-gray-900">KYC Verification</h2>
              <p className="text-gray-600 mt-1">Complete your KYC verification process</p>
            </div>
            
            <div className="p-6 text-center">
              <div className="max-w-md mx-auto">
                <Clock className="w-16 h-16 text-blue-600 mx-auto mb-4" />
                <h3 className="text-lg font-semibold text-gray-900 mb-2">KYC Verification in Progress</h3>
                <p className="text-gray-600 mb-6">
                  Your KYC verification is being processed. This usually takes 24-48 hours.
                </p>
                
                <div className="bg-gray-50 rounded-lg p-4 mb-6">
                  <div className="text-left space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">KYC ID:</span>
                      <span className="text-sm font-medium">{kycId}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">Status:</span>
                      <span className="px-2 py-1 text-xs bg-yellow-100 text-yellow-800 rounded-full">
                        PENDING
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">Submitted:</span>
                      <span className="text-sm font-medium">{formatDate(new Date())}</span>
                    </div>
                  </div>
                </div>
                
                <div className="space-y-3">
                  <button
                    onClick={checkKYCStatus}
                    className="w-full px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
                  >
                    Check Status
                  </button>
                  
                  <button
                    onClick={() => setStep(1)}
                    className="w-full px-4 py-2 border border-gray-300 text-gray-700 rounded-md hover:bg-gray-50 transition-colors"
                  >
                    Back to Details
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Step 3: KYC Complete */}
        {step === 3 && (
          <div className="bg-white rounded-lg shadow">
            <div className="p-6 border-b border-gray-200">
              <h2 className="text-xl font-semibold text-gray-900">KYC Verification Complete</h2>
            </div>
            
            <div className="p-6 text-center">
              <div className="max-w-md mx-auto">
                <CheckCircle className="w-16 h-16 text-green-600 mx-auto mb-4" />
                <h3 className="text-lg font-semibold text-gray-900 mb-2">KYC Verification Successful!</h3>
                <p className="text-gray-600 mb-6">
                  Your identity has been verified successfully. You can now proceed to setup eMandate for seamless investments.
                </p>
                
                <div className="bg-green-50 rounded-lg p-4 mb-6">
                  <div className="text-left space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">KYC ID:</span>
                      <span className="text-sm font-medium">{kycId}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">Status:</span>
                      <span className="px-2 py-1 text-xs bg-green-100 text-green-800 rounded-full">
                        COMPLETED
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">Completed:</span>
                      <span className="text-sm font-medium">{formatDate(new Date())}</span>
                    </div>
                  </div>
                </div>
                
                <button
                  onClick={() => setStep(4)}
                  className="w-full px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
                >
                  Setup eMandate
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Step 4: eMandate Setup */}
        {step === 4 && (
          <div className="bg-white rounded-lg shadow">
            <div className="p-6 border-b border-gray-200">
              <h2 className="text-xl font-semibold text-gray-900">eMandate Setup</h2>
              <p className="text-gray-600 mt-1">Setup automatic debit mandate for seamless SIP investments</p>
            </div>
            
            <div className="p-6">
              <div className="max-w-md mx-auto">
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">Bank Account Number *</label>
                    <input
                      type="text"
                      value={mandateData.accountNumber}
                      onChange={(e) => setMandateData({...mandateData, accountNumber: e.target.value})}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      placeholder="Enter account number"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">IFSC Code *</label>
                    <input
                      type="text"
                      value={mandateData.ifscCode}
                      onChange={(e) => setMandateData({...mandateData, ifscCode: e.target.value.toUpperCase()})}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      placeholder="SBIN0001234"
                      maxLength="11"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">Account Holder Name *</label>
                    <input
                      type="text"
                      value={mandateData.accountHolderName}
                      onChange={(e) => setMandateData({...mandateData, accountHolderName: e.target.value})}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      placeholder="Enter account holder name"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">Mandate Amount *</label>
                    <input
                      type="number"
                      value={mandateData.amount}
                      onChange={(e) => setMandateData({...mandateData, amount: Number(e.target.value)})}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      placeholder="Enter amount"
                      min="1000"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">Frequency *</label>
                    <select
                      value={mandateData.frequency}
                      onChange={(e) => setMandateData({...mandateData, frequency: e.target.value})}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    >
                      <option value="monthly">Monthly</option>
                      <option value="weekly">Weekly</option>
                      <option value="quarterly">Quarterly</option>
                    </select>
                  </div>
                </div>
                
                <div className="mt-8 space-y-3">
                  <button
                    onClick={handleMandateSubmit}
                    disabled={loading}
                    className="w-full px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors disabled:opacity-50"
                  >
                    {loading ? 'Processing...' : 'Setup eMandate'}
                  </button>
                  
                  <button
                    onClick={() => setStep(3)}
                    className="w-full px-4 py-2 border border-gray-300 text-gray-700 rounded-md hover:bg-gray-50 transition-colors"
                  >
                    Back
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Step 5: Complete */}
        {step === 5 && (
          <div className="bg-white rounded-lg shadow">
            <div className="p-6 border-b border-gray-200">
              <h2 className="text-xl font-semibold text-gray-900">Setup Complete</h2>
            </div>
            
            <div className="p-6 text-center">
              <div className="max-w-md mx-auto">
                <CheckCircle className="w-16 h-16 text-green-600 mx-auto mb-4" />
                <h3 className="text-lg font-semibold text-gray-900 mb-2">Congratulations!</h3>
                <p className="text-gray-600 mb-6">
                  Your KYC verification and eMandate setup are complete. You're now ready to start investing!
                </p>
                
                <div className="bg-green-50 rounded-lg p-4 mb-6">
                  <div className="text-left space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">KYC Status:</span>
                      <span className="px-2 py-1 text-xs bg-green-100 text-green-800 rounded-full">
                        COMPLETED
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">eMandate Status:</span>
                      <span className="px-2 py-1 text-xs bg-green-100 text-green-800 rounded-full">
                        ACTIVE
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">Mandate Amount:</span>
                      <span className="text-sm font-medium">₹{mandateData.amount.toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">Frequency:</span>
                      <span className="text-sm font-medium capitalize">{mandateData.frequency}</span>
                    </div>
                  </div>
                </div>
                
                <div className="space-y-3">
                  <button
                    onClick={() => window.location.href = '/dashboard'}
                    className="w-full px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
                  >
                    Go to Dashboard
                  </button>
                  
                  <button
                    onClick={() => window.location.href = '/invest'}
                    className="w-full px-4 py-2 border border-gray-300 text-gray-700 rounded-md hover:bg-gray-50 transition-colors"
                  >
                    Start Investing
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

export default KYCFlow; 