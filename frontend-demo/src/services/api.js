import axios from 'axios';
import toast from 'react-hot-toast';

// Create axios instance
const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:3000/api',
  timeout: 30000,
});

// Request interceptor to add auth token
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('authToken');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    const message = error.response?.data?.message || error.message || 'Something went wrong';
    
    // Handle specific error codes
    if (error.response?.status === 401) {
      localStorage.removeItem('authToken');
      localStorage.removeItem('user');
      window.location.href = '/login';
      toast.error('Session expired. Please login again.');
    } else if (error.response?.status === 403) {
      toast.error('Access denied. You don\'t have permission for this action.');
    } else if (error.response?.status === 429) {
      toast.error('Too many requests. Please try again later.');
    } else if (error.response?.status >= 500) {
      toast.error('Server error. Please try again later.');
    } else {
      toast.error(message);
    }
    
    return Promise.reject(error);
  }
);

// Auth APIs
export const authAPI = {
  register: (userData) => api.post('/auth/register', userData),
  login: (credentials) => api.post('/auth/login', credentials),
  logout: () => api.post('/auth/logout'),
  getProfile: () => api.get('/auth/profile'),
  updateProfile: (data) => api.put('/auth/profile', data),
  changePassword: (data) => api.put('/auth/change-password', data),
  forgotPassword: (email) => api.post('/auth/forgot-password', { email }),
  resetPassword: (data) => api.post('/auth/reset-password', data),
};

// Portfolio APIs
export const portfolioAPI = {
  getOverview: () => api.get('/portfolio/overview'),
  getFunds: (params) => api.get('/portfolio/funds', { params }),
  invest: (orderData) => api.post('/portfolio/invest', orderData),
  getOrderStatus: (orderId) => api.get(`/portfolio/orders/${orderId}`),
  redeem: (redemptionData) => api.post('/portfolio/redeem', redemptionData),
  getTransactions: (params) => api.get('/portfolio/transactions', { params }),
};

// AI & AGI APIs
export const aiAPI = {
  getPortfolioInsights: () => api.get('/ai/portfolio-insights'),
  getRecommendations: (userId) => api.get(`/agi/recommendations/${userId}`),
  initializeAGI: (userId) => api.post('/agi/initialize', { userId }),
  getPredictions: (params) => api.get('/agi/predictions', { params }),
};

// KYC & Compliance APIs
export const kycAPI = {
  initiateKYC: (kycData) => api.post('/kyc/initiate', kycData),
  getKYCStatus: (kycId) => api.get(`/kyc/status/${kycId}`),
  setupEMandate: (mandateData) => api.post('/kyc/emandate', mandateData),
};

// Payment APIs
export const paymentAPI = {
  createPaymentIntent: (data) => api.post('/payment/create-intent', data),
  getPaymentStatus: (paymentId) => api.get(`/payment/status/${paymentId}`),
};

// Social & Gamification APIs
export const socialAPI = {
  getLeaderboard: (params) => api.get('/social/leaderboard', { params }),
  sharePortfolio: (data) => api.post('/social/share-portfolio', data),
  getAchievements: () => api.get('/social/achievements'),
};

// Learning APIs
export const learningAPI = {
  getModules: (params) => api.get('/learning/modules', { params }),
  startSession: (moduleId) => api.post('/learning/start-session', { moduleId }),
  submitAnswer: (data) => api.post('/learning/submit-answer', data),
};

// Analytics APIs
export const analyticsAPI = {
  getDashboardData: () => api.get('/dashboard/overview'),
  getPerformance: (params) => api.get('/analytics/performance', { params }),
};

// Voice & Chat APIs
export const voiceAPI = {
  startSession: (data) => api.post('/voice/start-session', data),
  sendMessage: (data) => api.post('/voice/message', data),
};

// Regional Language APIs
export const regionalAPI = {
  getContent: (params) => api.get('/regional/content', { params }),
  translate: (data) => api.post('/regional/translate', data),
};

// BSE Star MF APIs (Demo)
export const bseStarMFAPI = {
  healthCheck: () => api.get('/bse-star-mf/health'),
  createClient: (clientData) => api.post('/bse-star-mf/client', clientData),
  getSchemes: (params) => api.get('/bse-star-mf/schemes', { params }),
  placeOrder: (orderData) => api.post('/bse-star-mf/order/lumpsum', orderData),
  getOrderStatus: (orderId) => api.get(`/bse-star-mf/order/status/${orderId}`),
  placeRedemption: (redemptionData) => api.post('/bse-star-mf/order/redemption', redemptionData),
  getTransactions: (params) => api.get('/bse-star-mf/report/transactions', { params }),
  getHoldings: (params) => api.get('/bse-star-mf/report/holdings', { params }),
  setupEMandate: (mandateData) => api.post('/bse-star-mf/emandate/setup', mandateData),
};

// Digio APIs (Demo)
export const digioAPI = {
  healthCheck: () => api.get('/digio/health'),
  initiateKYC: (kycData) => api.post('/digio/kyc/initiate', kycData),
  getKYCStatus: (kycId) => api.get(`/digio/kyc/status/${kycId}`),
  setupEMandate: (mandateData) => api.post('/digio/emandate/setup', mandateData),
  verifyPAN: (panNumber) => api.post('/digio/pan/verify', { panNumber }),
  pullCKYC: (ckycData) => api.post('/digio/ckyc/pull', ckycData),
  initiateESign: (esignData) => api.post('/digio/esign/initiate', esignData),
};

// Utility functions
export const apiUtils = {
  // Handle API responses
  handleResponse: (response) => {
    if (response.data.success) {
      return response.data.data;
    } else {
      throw new Error(response.data.message || 'API request failed');
    }
  },

  // Handle API errors
  handleError: (error) => {
    const message = error.response?.data?.message || error.message || 'Something went wrong';
    toast.error(message);
    throw error;
  },

  // Make API call with error handling
  apiCall: async (apiFunction, ...args) => {
    try {
      const response = await apiFunction(...args);
      return apiUtils.handleResponse(response);
    } catch (error) {
      return apiUtils.handleError(error);
    }
  },
};

export default api; 