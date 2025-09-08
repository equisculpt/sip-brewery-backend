import React, { createContext, useContext, useState, useEffect } from 'react';
import { authAPI } from '../services/api';
import toast from 'react-hot-toast';

const AuthContext = createContext();

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  // Check if user is logged in on app start
  useEffect(() => {
    const token = localStorage.getItem('authToken');
    const savedUser = localStorage.getItem('user');
    
    if (token && savedUser) {
      try {
        setUser(JSON.parse(savedUser));
        setIsAuthenticated(true);
      } catch (error) {
        localStorage.removeItem('authToken');
        localStorage.removeItem('user');
      }
    }
    setLoading(false);
  }, []);

  // Login function
  const login = async (credentials) => {
    try {
      const response = await authAPI.login(credentials);
      const { token, user: userData } = response.data.data;
      
      localStorage.setItem('authToken', token);
      localStorage.setItem('user', JSON.stringify(userData));
      
      setUser(userData);
      setIsAuthenticated(true);
      
      toast.success('Login successful!');
      return { success: true };
    } catch (error) {
      toast.error('Login failed. Please check your credentials.');
      return { success: false, error: error.message };
    }
  };

  // Register function
  const register = async (userData) => {
    try {
      const response = await authAPI.register(userData);
      const { token, user: newUser } = response.data.data;
      
      localStorage.setItem('authToken', token);
      localStorage.setItem('user', JSON.stringify(newUser));
      
      setUser(newUser);
      setIsAuthenticated(true);
      
      toast.success('Registration successful! Welcome to SipBrewery!');
      return { success: true };
    } catch (error) {
      toast.error('Registration failed. Please try again.');
      return { success: false, error: error.message };
    }
  };

  // Logout function
  const logout = async () => {
    try {
      await authAPI.logout();
    } catch (error) {
      // Continue with logout even if API call fails
    }
    
    localStorage.removeItem('authToken');
    localStorage.removeItem('user');
    
    setUser(null);
    setIsAuthenticated(false);
    
    toast.success('Logged out successfully');
  };

  // Update profile function
  const updateProfile = async (profileData) => {
    try {
      const response = await authAPI.updateProfile(profileData);
      const updatedUser = response.data.data;
      
      localStorage.setItem('user', JSON.stringify(updatedUser));
      setUser(updatedUser);
      
      toast.success('Profile updated successfully!');
      return { success: true };
    } catch (error) {
      toast.error('Failed to update profile. Please try again.');
      return { success: false, error: error.message };
    }
  };

  // Change password function
  const changePassword = async (passwordData) => {
    try {
      await authAPI.changePassword(passwordData);
      toast.success('Password changed successfully!');
      return { success: true };
    } catch (error) {
      toast.error('Failed to change password. Please try again.');
      return { success: false, error: error.message };
    }
  };

  // Forgot password function
  const forgotPassword = async (email) => {
    try {
      await authAPI.forgotPassword(email);
      toast.success('Password reset instructions sent to your email!');
      return { success: true };
    } catch (error) {
      toast.error('Failed to send reset instructions. Please try again.');
      return { success: false, error: error.message };
    }
  };

  // Reset password function
  const resetPassword = async (resetData) => {
    try {
      await authAPI.resetPassword(resetData);
      toast.success('Password reset successfully! Please login with your new password.');
      return { success: true };
    } catch (error) {
      toast.error('Failed to reset password. Please try again.');
      return { success: false, error: error.message };
    }
  };

  // Refresh user data
  const refreshUser = async () => {
    try {
      const response = await authAPI.getProfile();
      const userData = response.data.data;
      
      localStorage.setItem('user', JSON.stringify(userData));
      setUser(userData);
      
      return { success: true };
    } catch (error) {
      // If profile fetch fails, user might be logged out
      logout();
      return { success: false, error: error.message };
    }
  };

  const value = {
    user,
    loading,
    isAuthenticated,
    login,
    register,
    logout,
    updateProfile,
    changePassword,
    forgotPassword,
    resetPassword,
    refreshUser,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}; 