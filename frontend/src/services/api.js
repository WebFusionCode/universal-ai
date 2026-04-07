import axios from "axios";

const API_BASE_URL =
  process.env.REACT_APP_API_URL || "http://localhost:5000";

const API = axios.create({
  baseURL: API_BASE_URL,
});

API.interceptors.request.use((req) => {
  const token = localStorage.getItem("token");

  if (token) {
    req.headers.Authorization = `Bearer ${token}`;
  }

  return req;
});

API.interceptors.response.use(
  (res) => res,
  (err) => {
    if (err.response?.status === 401) {
      localStorage.removeItem("token");
      localStorage.removeItem("user_id");
      localStorage.removeItem("user_email");
      localStorage.removeItem("user_role");
      localStorage.removeItem("user_plan");
      window.location.href = "/login";
    }
    return Promise.reject(err);
  },
);

export default API;
