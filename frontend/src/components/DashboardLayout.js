import React, { useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import AIChat from './AIChat';

const navItems = [
  { path: '/dashboard', label: 'Dashboard', icon: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>
  )},
  { path: '/train', label: 'Train Model', icon: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg>
  )},
  { path: '/experiments', label: 'Experiments', icon: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M9 3h6v7l4 4v1H5v-1l4-4V3z"/><path d="M6 21h12"/></svg>
  )},
  { path: '/leaderboard', label: 'Leaderboard', icon: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M6 9H4.5a2.5 2.5 0 010-5H6M18 9h1.5a2.5 2.5 0 000-5H18M4 22h16M18 2H6v7a6 6 0 0012 0V2z"/></svg>
  )},
  { path: '/predict', label: 'Predict', icon: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><circle cx="12" cy="12" r="9"/><path d="M12 8v4l3 3"/></svg>
  )},
  { path: '/model-explain', label: 'Model Explain', icon: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20z"/><path d="M12 8v4M12 16h.01"/></svg>
  )},
  { path: '/insights', label: 'Insights', icon: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><line x1="12" y1="2" x2="12" y2="22"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>
  )},
  { path: '/image-ai', label: 'Image AI', icon: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="M21 15l-5-5L5 21"/></svg>
  )},
  { path: '/audio-ai', label: 'Audio AI', icon: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M12 1v22M17 5v14M7 9v6M19 2v20M5 15v6"/></svg>
  )},
  { path: '/video-ai', label: 'Video AI', icon: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><polygon points="23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/></svg>
  )},
  { path: '/adversarial-testing', label: 'Adversarial Test', icon: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z"/><path d="M12 6v6m0 4v.01"/></svg>
  )},
  { path: '/compiler', label: 'Code Compiler', icon: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>
  )},
  { path: '/teams', label: 'Teams', icon: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>
  )},
  { path: '/download', label: 'Downloads', icon: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
  )},
  { path: '/chatbot', label: 'AI Chatbot', icon: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
  )},
  { path: '/profile', label: 'Profile', icon: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>
  )},
  { path: '/admin', label: 'Admin', icon: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z"/><path d="M12 6a3 3 0 110 6 3 3 0 010-6zm4.02 7.02A6 6 0 0012 18a6 6 0 00-4.02-2.98"/></svg>
  )},
];

export default function DashboardLayout({ children, title }) {
  const location = useLocation();
  const navigate = useNavigate();
  const email = localStorage.getItem('email') || '';
  const [collapsed, setCollapsed] = useState(false);

  const logout = () => { localStorage.clear(); navigate('/login'); };

  return (
    <div className="h-screen flex flex-col bg-[#0a0a0a] overflow-hidden">
      {/* Top Bar */}
      <div className="flex items-center justify-between px-6 h-12 border-b border-white/[.06] bg-[#0a0a0a] shrink-0 z-20">
        <div className="flex items-center gap-5">
          <button onClick={() => setCollapsed(!collapsed)} className="text-white/40 hover:text-white/70 transition-colors md:hidden">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M3 12h18M3 6h18M3 18h18"/></svg>
          </button>
          <Link to="/dashboard" className="font-display text-xs tracking-[0.2em] uppercase font-bold text-white">
            AutoML <span style={{ color: '#B7FF4A' }}>X</span>
          </Link>
          {title && (
            <>
              <span className="text-white/10">/</span>
              <span className="font-mono text-[10px] tracking-[0.15em] uppercase text-white/30">{title}</span>
            </>
          )}
        </div>
        <div className="flex items-center gap-4">
          <span className="font-mono text-[10px] text-white/25 tracking-wider uppercase hidden sm:block">{email}</span>
          <button onClick={logout} data-testid="logout-btn"
            className="font-mono text-[10px] tracking-[0.12em] uppercase text-white/30 hover:text-white/60 transition-colors px-3 py-1.5 border border-white/[.06] hover:border-white/[.12]">
            Logout
          </button>
        </div>
      </div>

      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <aside className={`${collapsed ? 'w-0 md:w-14' : 'w-52'} shrink-0 border-r border-white/[.06] flex flex-col py-4 overflow-hidden transition-all duration-300`}>
          <nav className="flex-1 flex flex-col gap-0.5 px-2">
            {navItems.map((item) => {
              const active = location.pathname === item.path;
              return (
                <Link key={item.path} to={item.path} data-testid={`nav-${item.label.toLowerCase().replace(/\s/g, '-')}`}
                  className={`flex items-center gap-3 px-3 py-2.5 transition-all duration-200 ${active
                    ? 'bg-white/[.05] text-[#B7FF4A] border-l-2 border-[#B7FF4A]'
                    : 'text-white/35 hover:text-white/60 hover:bg-white/[.02] border-l-2 border-transparent'
                  }`}>
                  <span className="shrink-0">{item.icon}</span>
                  {!collapsed && <span className="font-mono text-[10px] tracking-[0.1em] uppercase truncate">{item.label}</span>}
                </Link>
              );
            })}
          </nav>
        </aside>

        {/* Main Content */}
        <main className="flex-1 overflow-y-auto p-6 md:p-8" style={{ scrollbarGutter: 'stable' }}>
          <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3 }}>
            {children}
          </motion.div>
        </main>
      </div>

      <AIChat />
    </div>
  );
}
