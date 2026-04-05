import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import './App.css';
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

// New Advanced Features
import Insights from './pages/Insights';
import ModelExplain from './pages/ModelExplain';
import AdversarialTesting from './pages/AdversarialTesting';
import GenerativeAI from './pages/GenerativeAI';

function ProtectedRoute({ children }) {
  const token = localStorage.getItem('token');
  if (!token) return <Navigate to="/login" replace />;
  return children;
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/login" element={<Login />} />
        <Route path="/signup" element={<Signup />} />
        
        {/* Core ML */}
        <Route path="/dashboard" element={<ProtectedRoute><Dashboard /></ProtectedRoute>} />
        <Route path="/train" element={<ProtectedRoute><Train /></ProtectedRoute>} />
        <Route path="/experiments" element={<ProtectedRoute><Experiments /></ProtectedRoute>} />
        <Route path="/leaderboard" element={<ProtectedRoute><Leaderboard /></ProtectedRoute>} />
        <Route path="/predict" element={<ProtectedRoute><Predict /></ProtectedRoute>} />
        <Route path="/download" element={<ProtectedRoute><Download /></ProtectedRoute>} />
        <Route path="/profile" element={<ProtectedRoute><Profile /></ProtectedRoute>} />
        
        {/* Advanced Features */}
        <Route path="/insights" element={<ProtectedRoute><Insights /></ProtectedRoute>} />
        <Route path="/explainable-ai" element={<ProtectedRoute><ModelExplain /></ProtectedRoute>} />
        <Route path="/adversarial-testing" element={<ProtectedRoute><AdversarialTesting /></ProtectedRoute>} />
        <Route path="/generative-ai" element={<ProtectedRoute><GenerativeAI /></ProtectedRoute>} />

        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}
