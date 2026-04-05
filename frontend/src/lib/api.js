import axios from 'axios';

const API_BASE = process.env.REACT_APP_BACKEND_URL || '';

const API = axios.create({
  baseURL: API_BASE,
});

API.interceptors.request.use((config) => {
  // Map frontend /api/* requests to backend /* routes
  if (config.url && config.url.startsWith('/api/')) {
    config.url = config.url.replace('/api/', '/');
  }

  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

API.interceptors.response.use(
  (res) => res,
  (err) => {
    if (err.response?.status === 401) {
      localStorage.removeItem('token');
      localStorage.removeItem('user_id');
      localStorage.removeItem('email');
      window.location.href = '/login';
    }
    return Promise.reject(err);
  }
);

export default API;
