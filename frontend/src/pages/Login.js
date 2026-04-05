import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import API from '../lib/api';

export default function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleLogin = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      const res = await API.post("/login", {
        email,
        password,
      });

      localStorage.setItem("token", res.data.access_token);
      localStorage.setItem("user_id", res.data.user_id);

      navigate("/dashboard");

    } catch (err) {
      setError(err.response?.data?.detail || "Login failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="relative min-h-screen flex items-center justify-center bg-[#0a0a0a] overflow-hidden">
      {/* Subtle orbs */}
      <div className="absolute top-[-20%] left-[-10%] w-[600px] h-[600px] rounded-full bg-[#B7FF4A]/5 blur-[150px] animate-float" />
      <div className="absolute bottom-[-20%] right-[-10%] w-[500px] h-[500px] rounded-full bg-[#6AA7FF]/5 blur-[130px] animate-float-delayed" />

      <motion.div initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6 }}
        className="relative z-10 w-full max-w-sm mx-4">
        <div className="border border-white/[.08] bg-[#111] p-8">
          <Link to="/" className="block text-center mb-8">
            <span className="font-display text-sm tracking-[0.2em] uppercase font-bold text-white">AutoML <span style={{ color: '#B7FF4A' }}>X</span></span>
          </Link>
          <h2 className="font-display text-xl font-bold uppercase tracking-tight text-white mb-1">Welcome Back</h2>
          <p className="font-mono text-[11px] text-white/40 tracking-wider uppercase mb-8">Sign in to your dashboard</p>

          {error && <div className="border border-[#FF5C7A]/20 bg-[#FF5C7A]/5 px-4 py-3 font-mono text-[11px] text-[#FF5C7A] mb-4">{error}</div>}

          <form onSubmit={handleLogin} className="space-y-5">
            <div>
              <label className="font-mono text-[10px] text-white/30 tracking-wider uppercase mb-1.5 block">Email</label>
              <input data-testid="login-email" type="email" value={email} onChange={(e) => setEmail(e.target.value)} required
                className="w-full px-4 py-3 bg-white/[.03] border border-white/[.08] text-white font-mono text-[13px] placeholder-white/20 focus:outline-none focus:border-[#B7FF4A]/40 transition-all"
                placeholder="you@example.com" />
            </div>
            <div>
              <label className="font-mono text-[10px] text-white/30 tracking-wider uppercase mb-1.5 block">Password</label>
              <input data-testid="login-password" type="password" value={password} onChange={(e) => setPassword(e.target.value)} required
                className="w-full px-4 py-3 bg-white/[.03] border border-white/[.08] text-white font-mono text-[13px] placeholder-white/20 focus:outline-none focus:border-[#B7FF4A]/40 transition-all"
                placeholder="Enter password" />
            </div>
            <button data-testid="login-submit" type="submit" disabled={loading}
              className="w-full py-3 bg-[#B7FF4A] text-[#0a0a0a] font-mono text-[11px] font-bold tracking-[0.1em] uppercase hover:bg-[#c8ff73] transition-all disabled:opacity-50">
              {loading ? 'Signing in...' : 'Sign In'}
            </button>
          </form>

          <div className="mt-8 text-center">
            <span className="font-mono text-[10px] text-white/30 tracking-wider uppercase">No account? </span>
            <Link to="/signup" className="font-mono text-[10px] text-[#B7FF4A] tracking-wider uppercase hover:text-[#c8ff73] transition-colors" data-testid="login-signup-link">Sign Up</Link>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
