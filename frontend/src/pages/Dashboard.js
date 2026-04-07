import React, { useState, useEffect, useCallback } from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import DashboardLayout from '../components/DashboardLayout';
import API from '../lib/api';

const fadeUp = { hidden: { opacity: 0, y: 20 }, visible: { opacity: 1, y: 0 } };

function StatCard({ label, value, num, color }) {
  return (
    <motion.div variants={fadeUp}
      className="border border-white/[.06] p-6 hover:border-white/[.12] transition-all duration-300 group cursor-default">
      <div className="flex items-start justify-between mb-4">
        <p className="font-mono text-[10px] tracking-[0.15em] uppercase text-white/30">{label}</p>
        <span className="font-mono text-[10px] text-white/15">{num}</span>
      </div>
      <p className="font-display text-3xl font-bold" style={{ color }}>{value}</p>
    </motion.div>
  );
}

export default function Dashboard() {
  const [stats, setStats] = useState({ experiments: 0, bestScore: '-', bestModel: '-' });
  const [recent, setRecent] = useState([]);

  const loadData = useCallback(async () => {
    try {
      const [expRes, insRes] = await Promise.all([
        API.get('/experiments').catch(() => ({ data: { experiments: [] } })),
        API.get('/api/insights').catch(() => ({ data: {} })),
      ]);
      const exps = expRes.data.experiments || [];
      setRecent(exps.slice(0, 5));
      setStats({
        experiments: exps.length,
        bestScore: insRes.data.best_score != null ? insRes.data.best_score.toFixed(4) : '-',
        bestModel: insRes.data.best_model || '-',
      });
    } catch (e) { console.error(e); }
  }, []);

  useEffect(() => { loadData(); }, [loadData]);

  return (
    <DashboardLayout title="Overview">
      <motion.div initial="hidden" animate="visible" variants={{ visible: { transition: { staggerChildren: 0.06 } } }}>
        <motion.div variants={fadeUp} className="mb-8">
          <h1 className="font-display text-2xl font-bold uppercase tracking-tight">Dashboard</h1>
          <p className="font-mono text-[11px] text-white/30 tracking-wider uppercase mt-1">Your ML workspace overview</p>
        </motion.div>

        {/* Stats */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 mb-8">
          <StatCard label="Total Experiments" value={stats.experiments} num="01" color="#6AA7FF" />
          <StatCard label="Best Score" value={stats.bestScore} num="02" color="#B7FF4A" />
          <StatCard label="Best Model" value={stats.bestModel} num="03" color="#FFCC66" />
          <StatCard label="Status" value="Ready" num="04" color="#5b9ea6" />
        </div>

        {/* Quick Actions */}
        <motion.div variants={fadeUp} className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-8">
          {[
            { to: '/train', icon: '\u26a1', label: 'Train New Model', color: '#B7FF4A' },
            { to: '/predict', icon: '\ud83d\udd2e', label: 'Make Predictions', color: '#6AA7FF' },
            { to: '/download', icon: '\ud83d\udce6', label: 'Download Models', color: '#5b9ea6' },
          ].map((action, i) => (
            <Link key={i} to={action.to} data-testid={`quick-${action.label.split(' ')[0].toLowerCase()}`}
              className="border border-white/[.06] p-5 text-center hover:border-white/[.12] hover:bg-white/[.02] transition-all duration-300 group">
              <div className="text-2xl mb-2">{action.icon}</div>
              <div className="font-mono text-[10px] tracking-[0.12em] uppercase text-white/40 group-hover:text-white/70 transition-colors" style={{ '--hover-color': action.color }}>
                {action.label}
              </div>
            </Link>
          ))}
        </motion.div>

        {/* Recent Experiments */}
        <motion.div variants={fadeUp} className="border border-white/[.06]">
          <div className="px-6 py-4 border-b border-white/[.06] flex justify-between items-center">
            <h3 className="font-mono text-[10px] tracking-[0.15em] uppercase text-white/40">Recent Experiments</h3>
            <Link to="/experiments" className="font-mono text-[10px] tracking-[0.12em] uppercase text-[#B7FF4A] hover:text-[#c8ff73] transition-colors">View All</Link>
          </div>
          {recent.length === 0 ? (
            <div className="p-10 text-center">
              <p className="font-mono text-[11px] text-white/25 tracking-wider uppercase">No experiments yet. Train your first model.</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-white/[.06]">
                    {['Model', 'Score', 'Type', 'Time'].map(h => (
                      <th key={h} className="px-6 py-3 text-left font-mono text-[10px] text-white/25 tracking-[0.15em] uppercase font-normal">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {recent.map((exp, i) => (
                    <tr key={i} className="border-b border-white/[.04] hover:bg-white/[.02] transition-colors">
                      <td className="px-6 py-3 font-mono text-[12px] text-white/70">{exp.best_model || exp.model_name || '-'}</td>
                      <td className="px-6 py-3 font-mono text-[12px] text-[#B7FF4A]">{exp.score != null ? Number(exp.score).toFixed(4) : '-'}</td>
                      <td className="px-6 py-3 font-mono text-[11px] text-white/40">{exp.problem_type || '-'}</td>
                      <td className="px-6 py-3 font-mono text-[10px] text-white/25">{exp.time || exp.timestamp || '-'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </motion.div>
      </motion.div>
    </DashboardLayout>
  );
}
