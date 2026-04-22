import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import DashboardLayout from '../components/DashboardLayout';
import { 
  Trophy, History, BarChart3, Database, Image as ImageIcon, 
  Clock, ArrowRight, Zap, RefreshCw
} from 'lucide-react';
import API from '../lib/api';

const fadeUp = { hidden: { opacity: 0, y: 20 }, visible: { opacity: 1, y: 0 } };

const safeNumber = (val, decimals = 4) =>
  typeof val === 'number' ? val.toFixed(decimals) : '-';

export default function Experiments() {
  const [experiments, setExperiments] = useState([]);
  const [leaderboard, setLeaderboard] = useState([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('all'); // all, classification, regression, image, time_series
  const [lastRefreshed, setLastRefreshed] = useState(new Date());

  const isRegressionTab = activeTab === 'regression';
  const scoreLabel = isRegressionTab ? 'R² Score' : 'Score';
  const lossLabel = isRegressionTab ? 'MSE Loss' : 'Loss';

  const loadData = useCallback(async () => {
    try {
      if (activeTab === 'all') {
        const res = await API.get('/experiments');
        setExperiments(Array.isArray(res.data) ? res.data : []);
      } else {
        const res = await API.get(`/leaderboard/${activeTab}`);
        setLeaderboard(Array.isArray(res.data) ? res.data : []);
      }
      setLastRefreshed(new Date());
    } catch (e) {
      console.error("Fetch failed:", e);
    } finally {
      setLoading(false);
    }
  }, [activeTab]);

  // Initial load when tab changes
  useEffect(() => {
    setLoading(true);
    loadData();
  }, [loadData]);

  // Polling every 5 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      loadData();
    }, 5000);
    return () => clearInterval(interval);
  }, [loadData]);

  const tabs = [
    { id: 'all', label: 'Recent Runs', icon: History },
    { id: 'classification', label: 'Classification', icon: Database },
    { id: 'regression', label: 'Regression', icon: BarChart3 },
    { id: 'image', label: 'Image Models', icon: ImageIcon },
    { id: 'time_series', label: 'Time-Series', icon: Clock },
  ];

  return (
    <DashboardLayout>
      <div className="space-y-6">
        
        {/* Header */}
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
          <div>
            <h1 className="font-display text-2xl font-bold uppercase tracking-tight text-white">Experiments & Leaderboards</h1>
            <div className="flex items-center gap-3 mt-1">
              <p className="font-mono text-[11px] text-white/30 tracking-wider uppercase">
                {activeTab === 'all' ? `${experiments.length} runs recorded` : `Top results for ${activeTab}`}
              </p>
              <div className="h-1 w-1 rounded-full bg-white/10" />
              <p className="font-mono text-[10px] text-[#B7FF4A]/60 uppercase flex items-center gap-1.5 animate-pulse">
                <RefreshCw size={10} /> Realtime Polling Active (5s)
              </p>
            </div>
          </div>
          <div className="font-mono text-[10px] text-white/20 uppercase">
            Updated: {lastRefreshed.toLocaleTimeString()}
          </div>
        </div>

        {/* Tabs */}
        <div className="flex flex-wrap items-center gap-2 border-b border-white/[.06] pb-1">
          {tabs.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setActiveTab(id)}
              className={`flex items-center gap-2 px-4 py-3 font-mono text-[11px] font-bold uppercase tracking-wider transition-all border-b-2 ${
                activeTab === id 
                ? 'text-[#B7FF4A] border-[#B7FF4A] bg-[#B7FF4A]/5' 
                : 'text-white/40 border-transparent hover:text-white/60 hover:bg-white/[.02]'
              }`}
            >
              <Icon size={14} />
              {label}
            </button>
          ))}
        </div>

        {/* Content */}
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial="hidden"
            animate="visible"
            exit="hidden"
            variants={fadeUp}
            transition={{ duration: 0.2 }}
          >
            {loading && activeTab === 'all' && experiments.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-32 space-y-4">
                <div className="w-8 h-8 border-2 border-[#B7FF4A] border-t-transparent rounded-full animate-spin" />
                <p className="font-mono text-[10px] text-white/40 uppercase tracking-widest">Calibrating History...</p>
              </div>
            ) : (activeTab === 'all' ? experiments : leaderboard).length === 0 ? (
              <div className="border border-white/[.06] bg-[#111] p-20 text-center">
                <Zap size={32} className="text-white/10 mx-auto mb-4" />
                <p className="font-display text-sm font-bold uppercase text-white/40">No data records found</p>
                <p className="font-mono text-[11px] text-white/20 mt-2 uppercase">Training models populate various leaderboards automatically.</p>
              </div>
            ) : (
              <div className="border border-white/[.08] bg-[#111] overflow-hidden">
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
	                      <tr className="border-b border-white/[.08] bg-white/[.02]">
                        <th className="px-6 py-4 text-left font-mono text-[10px] text-white/30 uppercase tracking-[0.15em] font-normal">
                          {activeTab === 'all' ? 'Rank' : 'Pos'}
                        </th>
                        <th className="px-6 py-4 text-left font-mono text-[10px] text-white/30 uppercase tracking-[0.15em] font-normal">Best Model</th>
	                        <th className="px-6 py-4 text-left font-mono text-[10px] text-white/30 uppercase tracking-[0.15em] font-normal">{scoreLabel}</th>
	                        <th className="px-6 py-4 text-left font-mono text-[10px] text-white/30 uppercase tracking-[0.15em] font-normal">{lossLabel}</th>
                        <th className="px-6 py-4 text-left font-mono text-[10px] text-white/30 uppercase tracking-[0.15em] font-normal">Metrics</th>
                        <th className="px-6 py-4 text-left font-mono text-[10px] text-white/30 uppercase tracking-[0.15em] font-normal">Dataset</th>
                        <th className="px-6 py-4 text-left font-mono text-[10px] text-white/30 uppercase tracking-[0.15em] font-normal">Timestamp</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(activeTab === 'all' ? experiments : leaderboard).map((exp, i) => (
                        <tr 
                          key={exp.id || i} 
                          className={`border-b border-white/[.04] hover:bg-white/[.02] transition-colors ${i === 0 && activeTab !== 'all' ? 'bg-[#B7FF4A]/[0.03]' : ''}`}
                        >
                          <td className="px-6 py-4">
                            {activeTab === 'all' ? (
                              <span className="font-mono text-[11px] text-white/20">#{experiments.length - i}</span>
                            ) : (
                              <span className="flex items-center gap-2">
                                {i === 0 ? <Trophy size={14} className="text-[#B7FF4A]" /> : null}
                                <span className={`font-mono text-xs ${i === 0 ? 'text-[#B7FF4A] font-bold' : 'text-white/40'}`}>
                                  {i + 1}
                                </span>
                              </span>
                            )}
                          </td>
                          <td className="px-6 py-4">
                            <div className="flex flex-col">
                              <span className="font-display text-[13px] font-bold text-white uppercase tracking-tight">
                                {exp.best_model || exp.model_name || '-'}
                              </span>
                              <span className="font-mono text-[9px] text-[#666] uppercase mt-0.5">
                                {exp.problem_type} • {exp.model_version || 'v1.0'}
                              </span>
                            </div>
                          </td>
                          <td className="px-6 py-4">
                            <span className="font-mono text-[14px] font-bold text-[#B7FF4A]">
                              {safeNumber(exp.score)}
                            </span>
                          </td>
                          <td className="px-6 py-4">
                            <span className="font-mono text-[14px] font-bold text-[#FF6B6B]">
                              {safeNumber(exp.loss)}
                            </span>
                          </td>
                          <td className="px-6 py-4">
                            <div className="flex gap-4">
                              {exp.metrics && Object.entries(exp.metrics).slice(0, 2).map(([key, val]) => (
                                <div key={key}>
                                  <div className="font-mono text-[8px] text-white/20 uppercase">{key}</div>
                                  <div className="font-mono text-[10px] text-white/60">{safeNumber(val, 3)}</div>
                                </div>
                              ))}
                            </div>
                          </td>
                          <td className="px-6 py-4">
                            <div className="flex items-center gap-2 group cursor-help">
                              <Database size={12} className="text-white/20 group-hover:text-[#B7FF4A] transition-colors" />
                              <div className="flex flex-col">
                                <span className="font-mono text-[11px] text-white/60 truncate max-w-[120px]">
                                  {exp.dataset || 'uploaded_data'}
                                </span>
                                <span className="font-mono text-[9px] text-white/20 uppercase">
                                  {exp.rows || '-'} rows
                                </span>
                              </div>
                            </div>
                          </td>
                          <td className="px-6 py-4 text-right">
                            <div className="flex flex-col items-end">
                              <span className="font-mono text-[10px] text-white/40">
                                {exp.created_at ? new Date(exp.created_at).toLocaleDateString() : '-'}
                              </span>
                              <span className="font-mono text-[9px] text-white/20 uppercase mt-0.5">
                                {exp.created_at ? new Date(exp.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '-'}
                              </span>
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </motion.div>
        </AnimatePresence>
      </div>
    </DashboardLayout>
  );
}
