import React, { useEffect, useState } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import './App.css';

// Pages
import Landing from './pages/Landing';
import Login from './pages/Login';
import Signup from './pages/Signup';
import Dashboard from './pages/Dashboard';
import Train from './pages/Train';
import Experiments from './pages/Experiments';
import Leaderboard from './pages/Leaderboard';
import Predict from './pages/Predict';
import Download from './pages/Download';
import Profile from './pages/Profile';
import Admin from './pages/Admin';
import ModelExplain from './pages/ModelExplain';
import TestModel from './pages/TestModel';
import ImageAI from './pages/ImageAI';
import AudioAI from './pages/AudioAI';
import VideoAI from './pages/VideoAI';
import AdversarialTesting from './pages/AdversarialTesting';
import Compiler from './pages/Compiler';
import Insights from './pages/Insights';
import Teams from './pages/Teams';
import Pricing from './pages/Pricing';
import Chatbot from './pages/Chatbot';

function ProtectedRoute({ children }) {
  const token = localStorage.getItem('token');
  if (!token) return <Navigate to="/login" replace />;
  return children;
}

export default function App() {
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      setLoading(false);
    }, 1000);

    return () => {
      window.clearTimeout(timer);
    };
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-[#0a0a0a]">
        <div className="text-center">
          <div className="inline-block">
            <div className="w-12 h-12 border-4 border-[#B7FF4A]/20 border-t-[#B7FF4A] rounded-full animate-spin"></div>
          </div>
          <p className="mt-4 text-white/60">Loading...</p>
        </div>
      </div>
    );
  }

  return (
    <BrowserRouter>
      <Routes>
        {/* Public Routes */}
        <Route path="/" element={<Landing />} />
        <Route path="/login" element={<Login />} />
        <Route path="/signup" element={<Signup />} />
        <Route path="/pricing" element={<Pricing />} />

        {/* Protected Routes */}
        <Route path="/dashboard" element={<ProtectedRoute><Dashboard /></ProtectedRoute>} />
        <Route path="/train" element={<ProtectedRoute><Train /></ProtectedRoute>} />
        <Route path="/experiments" element={<ProtectedRoute><Experiments /></ProtectedRoute>} />
        <Route path="/leaderboard" element={<ProtectedRoute><Leaderboard /></ProtectedRoute>} />
        <Route path="/predict" element={<ProtectedRoute><Predict /></ProtectedRoute>} />
        <Route path="/download" element={<ProtectedRoute><Download /></ProtectedRoute>} />
        <Route path="/profile" element={<ProtectedRoute><Profile /></ProtectedRoute>} />
        <Route path="/model-explain" element={<ProtectedRoute><ModelExplain /></ProtectedRoute>} />
        <Route path="/test-model" element={<ProtectedRoute><TestModel /></ProtectedRoute>} />
        <Route path="/image-ai" element={<ProtectedRoute><ImageAI /></ProtectedRoute>} />
        <Route path="/audio-ai" element={<ProtectedRoute><AudioAI /></ProtectedRoute>} />
        <Route path="/video-ai" element={<ProtectedRoute><VideoAI /></ProtectedRoute>} />
        <Route path="/adversarial-testing" element={<ProtectedRoute><AdversarialTesting /></ProtectedRoute>} />
        <Route path="/compiler" element={<ProtectedRoute><Compiler /></ProtectedRoute>} />
        <Route path="/insights" element={<ProtectedRoute><Insights /></ProtectedRoute>} />
        <Route path="/teams" element={<ProtectedRoute><Teams /></ProtectedRoute>} />
        <Route path="/admin" element={<ProtectedRoute><Admin /></ProtectedRoute>} />
        <Route path="/chatbot" element={<ProtectedRoute><Chatbot /></ProtectedRoute>} />

        {/* Catch all */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}
