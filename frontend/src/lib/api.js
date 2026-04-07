import axios from 'axios';

const API = axios.create({
  baseURL: process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000',
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor to add auth token
API.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor for error handling
API.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token');
      localStorage.removeItem('user_id');
      localStorage.removeItem('email');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export default API;
