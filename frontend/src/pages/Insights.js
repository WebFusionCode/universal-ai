import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import DashboardLayout from '../components/DashboardLayout';
import API from '../lib/api';
import { BrainCircuit, LineChart, Target, Zap } from 'lucide-react';

const fadeUp = { hidden: { opacity: 0, y: 20 }, visible: { opacity: 1, y: 0 } };

export default function Insights() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    const fetchInsights = async () => {
      try {
        const res = await API.get('/api/insights');
        setData(res.data);
      } catch (err) {
        setError('Failed to fetch AI insights. Train a model first.');
      } finally {
        setLoading(false);
      }
    };
    fetchInsights();
  }, []);

  return (
    <DashboardLayout title="AI Insights">
      <motion.div initial="hidden" animate="visible" variants={{ visible: { transition: { staggerChildren: 0.06 } } }}>
        <motion.div variants={fadeUp} className="mb-8">
          <h1 className="font-display text-2xl font-bold uppercase tracking-tight">AI Insights</h1>
          <p className="font-mono text-[11px] text-white/30 tracking-wider uppercase mt-1">Automated analysis of your dataset</p>
        </motion.div>

        {loading ? (
          <div className="flex justify-center p-20">
            <div className="w-8 h-8 border-2 border-[#B7FF4A] border-t-transparent rounded-full animate-spin" />
          </div>
        ) : error ? (
          <div className="border border-[#FF5C7A]/20 bg-[#FF5C7A]/5 px-6 py-4 font-mono text-[11px] text-[#FF5C7A]">{error}</div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            
            {/* Health Score */}
            <motion.div variants={fadeUp} className="border border-white/[.06] bg-white/[.02] p-6">
              <div className="flex items-center gap-3 mb-6">
                <Target className="w-5 h-5 text-[#B7FF4A]" />
                <h3 className="font-mono text-[11px] text-white/40 tracking-[0.15em] uppercase">Dataset Health</h3>
              </div>
              <div className="flex items-end gap-2 mb-2">
                <span className="font-display text-5xl font-bold text-white">{data?.health_score || 'N/A'}</span>
                <span className="font-mono text-[11px] text-white/30 mb-2">/100</span>
              </div>
              <div className="h-1 bg-white/[.06] mt-4">
                <div className="h-full bg-[#B7FF4A]" style={{ width: `${data?.health_score || 0}%` }} />
              </div>
            </motion.div>

            {/* Quick Stats */}
            <motion.div variants={fadeUp} className="border border-white/[.06] bg-white/[.02] p-6">
              <div className="flex items-center gap-3 mb-6">
                <LineChart className="w-5 h-5 text-[#6AA7FF]" />
                <h3 className="font-mono text-[11px] text-white/40 tracking-[0.15em] uppercase">Data Shape</h3>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="font-mono text-[10px] text-white/30 tracking-wider uppercase mb-1">Total Rows</p>
                  <p className="font-display text-2xl font-bold">{data?.total_rows || '0'}</p>
                </div>
                <div>
                  <p className="font-mono text-[10px] text-white/30 tracking-wider uppercase mb-1">Features</p>
                  <p className="font-display text-2xl font-bold">{data?.total_columns || '0'}</p>
                </div>
                <div>
                  <p className="font-mono text-[10px] text-white/30 tracking-wider uppercase mb-1">Missing Vals</p>
                  <p className="font-display text-xl font-bold text-[#FF5C7A]">{data?.missing_values || '0'}</p>
                </div>
                <div>
                  <p className="font-mono text-[10px] text-white/30 tracking-wider uppercase mb-1">Duplicates</p>
                  <p className="font-display text-xl font-bold text-[#FFCC66]">{data?.duplicates || '0'}</p>
                </div>
              </div>
            </motion.div>

            {/* AI Recommendations */}
            <motion.div variants={fadeUp} className="md:col-span-2 border border-white/[.06] bg-white/[.02] p-6">
              <div className="flex items-center gap-3 mb-6">
                <BrainCircuit className="w-5 h-5 text-[#A855F7]" />
                <h3 className="font-mono text-[11px] text-white/40 tracking-[0.15em] uppercase">AI Recommendations</h3>
              </div>
              <div className="space-y-4">
                {data?.recommendations?.length > 0 ? (
                  data.recommendations.map((rec, i) => (
                    <div key={i} className="flex gap-4 p-4 border border-white/[.04] bg-[#050505]">
                      <Zap className="w-4 h-4 text-[#FFCC66] shrink-0 mt-0.5" />
                      <div>
                        <p className="font-mono text-[12px] text-white/80 leading-relaxed">{rec}</p>
                      </div>
                    </div>
                  ))
                ) : (
                  <p className="font-mono text-[11px] text-white/30">No critical anomalies detected. Dataset is optimal.</p>
                )}
              </div>
            </motion.div>
            
          </div>
        )}
      </motion.div>
    </DashboardLayout>
  );
}
