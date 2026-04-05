import React, { useState, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import DashboardLayout from '../components/DashboardLayout';
import API from '../lib/api';

const fadeUp = { hidden: { opacity: 0, y: 20 }, visible: { opacity: 1, y: 0 } };

export default function Leaderboard() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    try { const res = await API.get('/leaderboard'); setData(res.data); } catch (e) {} finally { setLoading(false); }
  }, []);

  useEffect(() => { load(); }, [load]);

  return (
    <DashboardLayout title="Leaderboard">
      <motion.div initial="hidden" animate="visible" variants={{ visible: { transition: { staggerChildren: 0.06 } } }}>
        <motion.div variants={fadeUp} className="mb-8">
          <h1 className="font-display text-2xl font-bold uppercase tracking-tight">Leaderboard</h1>
          <p className="font-mono text-[11px] text-white/30 tracking-wider uppercase mt-1">Model performance rankings</p>
        </motion.div>

        {loading ? (
          <div className="flex justify-center py-20"><div className="w-6 h-6 border-2 border-[#B7FF4A] border-t-transparent rounded-full animate-spin" /></div>
        ) : !data || data.error || !data.models?.length ? (
          <motion.div variants={fadeUp} className="border border-white/[.06] p-16 text-center">
            <p className="font-mono text-[11px] text-white/25 tracking-wider uppercase">{data?.error || 'No leaderboard data. Train a model first.'}</p>
          </motion.div>
        ) : (
          <>
            <motion.div variants={fadeUp} className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-6">
              {[
                { label: 'Best Model', value: data.best_model, color: '#B7FF4A' },
                { label: 'Total Models', value: data.total_models, color: '#6AA7FF' },
                { label: 'Problem Type', value: data.problem_type || '-', color: '#FFCC66' },
              ].map((s, i) => (
                <div key={i} className="border border-white/[.06] p-5">
                  <p className="font-mono text-[10px] text-white/25 tracking-wider uppercase mb-2">{s.label}</p>
                  <p className="font-display text-lg font-bold" style={{ color: s.color }}>{s.value}</p>
                </div>
              ))}
            </motion.div>

            <motion.div variants={fadeUp} className="border border-white/[.06]">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-white/[.08]">
                    {['Rank', 'Model', 'Score', 'Time (s)'].map(h => (
                      <th key={h} className="px-6 py-3 text-left font-mono text-[10px] text-white/25 tracking-[0.15em] uppercase font-normal">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {data.models.map((m, i) => (
                    <motion.tr key={i} initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.04 }}
                      className={`border-b border-white/[.04] hover:bg-white/[.02] transition-colors ${m.rank === 1 ? 'bg-[#B7FF4A]/[.03]' : ''}`}>
                      <td className="px-6 py-4 font-mono text-lg">{m.rank === 1 ? '\ud83e\udd47' : m.rank === 2 ? '\ud83e\udd48' : m.rank === 3 ? '\ud83e\udd49' : m.rank}</td>
                      <td className="px-6 py-4 font-mono text-[13px] text-white/70">{m.model}</td>
                      <td className="px-6 py-4">
                        <span className="font-mono text-[13px] text-[#B7FF4A] font-bold">{m.score?.toFixed(4)}</span>
                        {m.rank === 1 && <span className="ml-2 font-mono text-[9px] text-[#B7FF4A]/60 tracking-wider uppercase border border-[#B7FF4A]/20 px-1.5 py-0.5">Best</span>}
                      </td>
                      <td className="px-6 py-4 font-mono text-[12px] text-white/40">{m.time?.toFixed(2)}</td>
                    </motion.tr>
                  ))}
                </tbody>
              </table>
            </motion.div>
          </>
        )}
      </motion.div>
    </DashboardLayout>
  );
}
