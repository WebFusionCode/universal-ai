import React, { useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import AIChat from './AIChat';

const navItems = [
  "Dashboard",
  "Train Model",
  "Predict",
  "Leaderboard",
  "AI Insights",
  "Explainable AI",
  "Downloads",
  "Sec-Test",
  "Generative Studio",
  "Compiler",
  "Profile"
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
              const staticMap = {
                "Train Model": "/train",
                "Downloads": "/download",
                "Sec-Test": "/adversarial-testing",
                "Generative Studio": "/generative-ai",
                "AI Insights": "/insights",
                "Explainable AI": "/explainable-ai"
              };
              const mappedPath = staticMap[item] || `/${item.toLowerCase().replace(/\s/g, '-')}`;
              const active = location.pathname === mappedPath;
              return (
                <Link key={mappedPath} to={mappedPath} data-testid={`nav-${item.toLowerCase().replace(/\s/g, '-')}`}
                  className={`flex items-center gap-3 px-3 py-2.5 transition-all duration-200 ${active
                    ? 'bg-white/[.05] text-[#B7FF4A] border-l-2 border-[#B7FF4A]'
                    : 'text-white/35 hover:text-white/60 hover:bg-white/[.02] border-l-2 border-transparent'
                  }`}>
                  <span className="shrink-0"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><circle cx="12" cy="12" r="3"/></svg></span>
                  {!collapsed && <span className="font-mono text-[10px] tracking-[0.1em] uppercase truncate">{item}</span>}
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
