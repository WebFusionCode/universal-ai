import { BrowserRouter, Routes, Route } from "react-router-dom";

import Landing from "./pages/Landing";
import Login from "./pages/Login";
import Signup from "./pages/Signup";
import Dashboard from "./pages/Dashboard";
import Train from "./pages/Train";
import Leaderboard from "./pages/Leaderboard";
import DownloadCenter from "./pages/Download";
import Insights from "./pages/Insights";
import Predict from "./pages/Predict";
import ImageAI from "./pages/ImageAI";
import Experiments from "./pages/Experiments";
import ProtectedRoute from "./components/ProtectedRoute";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>

        {/* Public Routes */}
        <Route path="/" element={<Landing />} />
        <Route path="/login" element={<Login />} />
        <Route path="/signup" element={<Signup />} />

        {/* 🔐 Protected Routes */}
        <Route
          path="/train"
          element={
            <ProtectedRoute>
              <Train />
            </ProtectedRoute>
          }
        />

        <Route
          path="/dashboard"
          element={
            <ProtectedRoute>
              <Dashboard />
            </ProtectedRoute>
          }
        />

        <Route
          path="/predict"
          element={
            <ProtectedRoute>
              <Predict />
            </ProtectedRoute>
          }
        />

        <Route
          path="/leaderboard"
          element={
            <ProtectedRoute>
              <Leaderboard />
            </ProtectedRoute>
          }
        />

        <Route
          path="/insights"
          element={
            <ProtectedRoute>
              <Insights />
            </ProtectedRoute>
          }
        />

        <Route
          path="/download"
          element={
            <ProtectedRoute>
              <DownloadCenter />
            </ProtectedRoute>
          }
        />

        <Route
          path="/image-ai"
          element={
            <ProtectedRoute>
              <ImageAI />
            </ProtectedRoute>
          }
        />

        <Route
          path="/experiments"
          element={
            <ProtectedRoute>
              <Experiments />
            </ProtectedRoute>
          }
        />

      </Routes>
    </BrowserRouter>
  );
}
