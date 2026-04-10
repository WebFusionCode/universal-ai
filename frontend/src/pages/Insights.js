import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import DashboardLayout from '../components/DashboardLayout';
import { Brain, TrendingUp, Trophy, BarChart3, RefreshCw } from 'lucide-react';
import API from '../lib/api';

const fadeUp = { hidden: { opacity: 0, y: 20 }, visible: { opacity: 1, y: 0 } };

const safeNumber = (val, d = 4) => (typeof val === 'number' ? val.toFixed(d) : '—');

export default function Insights() {
  const [summary, setSummary] = useState(null);
  const [aiInsights, setAiInsights] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const load = async () => {
    setLoading(true); setError('');
    try {
      const [summaryRes, aiRes] = await Promise.all([
        API.get('/insights').catch(() => ({ data: null })),
        API.get('/ai-insights').catch(() => ({ data: { insights: [] } }))
      ]);
      setSummary(summaryRes.data?.error ? null : summaryRes.data);
      setAiInsights(aiRes.data?.insights || []);
    } catch (e) {
      setError('Failed to load insights');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, []);

  const stats = summary ? [
    { label: 'Best Model', value: summary.best_model || '—', icon: Trophy, color: '#B7FF4A' },
    { label: 'Best Score', value: safeNumber(summary.best_score), icon: TrendingUp, color: '#6AA7FF' },
    { label: 'Total Experiments', value: summary.total_experiments ?? '—', icon: BarChart3, color: '#FF6B9D' },
    { label: 'Avg Score', value: safeNumber(summary.average_score), icon: Brain, color: '#4FD1C5' }
  ] : [];

  return (
    <DashboardLayout>
      <motion.div initial="hidden" animate="visible" variants={{ visible: { transition: { staggerChildren: 0.07 } } }}>

        {/* Header */}
        <motion.div variants={fadeUp} className="flex items-center justify-between mb-8">
          <div>
            <h1 className="font-display text-2xl font-bold uppercase tracking-tight text-white">Insights</h1>
            <p className="font-mono text-[11px] text-white/40 tracking-wider uppercase mt-1">
              AI-driven analytics across all your experiments
            </p>
          </div>
          <button
            onClick={load}
            disabled={loading}
            className="flex items-center gap-2 px-4 py-2 border border-white/[.08] text-white/40 font-mono text-[10px] uppercase tracking-wider hover:text-white hover:border-white/[.20] transition-all disabled:opacity-40"
          >
            <RefreshCw size={12} className={loading ? 'animate-spin' : ''} /> Refresh
          </button>
        </motion.div>

        {loading ? (
          <div className="flex justify-center py-20">
            <div className="w-6 h-6 border-2 border-[#B7FF4A] border-t-transparent rounded-full animate-spin" />
          </div>
        ) : error ? (
          <motion.div variants={fadeUp} className="border border-[#FF5C7A]/20 bg-[#FF5C7A]/5 px-4 py-3 font-mono text-[11px] text-[#FF5C7A]">
            {error}
          </motion.div>
        ) : !summary ? (
          <motion.div variants={fadeUp} className="border border-white/[.06] p-16 text-center">
            <Brain size={32} className="text-white/10 mx-auto mb-4" />
            <p className="font-mono text-[11px] text-white/25 tracking-wider uppercase">
              No experiments yet. Train a model to unlock insights.
            </p>
          </motion.div>
        ) : (
          <>
            {/* Stats Grid */}
            <motion.div variants={fadeUp} className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
              {stats.map(({ label, value, icon: Icon, color }) => (
                <div key={label} className="border border-white/[.06] p-5 hover:border-white/[.12] transition-all">
                  <div className="flex items-center gap-2 mb-3">
                    <Icon size={13} style={{ color }} />
                    <p className="font-mono text-[10px] tracking-[0.15em] uppercase text-white/30">{label}</p>
                  </div>
                  <p className="font-display text-2xl font-bold" style={{ color }}>{value}</p>
                </div>
              ))}
            </motion.div>

            {/* AI Insights */}
            {aiInsights.length > 0 && (
              <motion.div variants={fadeUp} className="border border-white/[.06] p-6 mb-6">
                <h2 className="font-display text-sm font-bold uppercase text-white mb-5 flex items-center gap-2">
                  <Brain size={14} className="text-[#B7FF4A]" /> AI Recommendations
                </h2>
                <div className="space-y-3">
                  {aiInsights.map((ins, i) => (
                    <div key={i} className="flex items-start gap-3 py-2.5 border-b border-white/[.04] last:border-0">
                      <span className="text-[#B7FF4A] font-bold shrink-0 mt-0.5">→</span>
                      <p className="font-mono text-[11px] text-white/60 leading-relaxed">{ins}</p>
                    </div>
                  ))}
                </div>
              </motion.div>
            )}

            {/* Best version */}
            {summary.best_version && (
              <motion.div variants={fadeUp} className="border border-white/[.06] p-5 flex items-center gap-4">
                <Trophy size={16} className="text-[#B7FF4A] shrink-0" />
                <div>
                  <p className="font-mono text-[10px] text-white/30 uppercase tracking-wider mb-1">Best Model File</p>
                  <p className="font-mono text-[12px] text-white/60">{summary.best_version}</p>
                </div>
              </motion.div>
            )}
          </>
        )}
      </motion.div>
    </DashboardLayout>
  );
}
