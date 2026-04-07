import React, { useState, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import DashboardLayout from '../components/DashboardLayout';
import API from '../lib/api';

const fadeUp = { hidden: { opacity: 0, y: 20 }, visible: { opacity: 1, y: 0 } };

const safeNumber = (val) => 
  typeof val === 'number' ? val.toFixed(4) : '-';

export default function Experiments() {
  const [experiments, setExperiments] = useState([]);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    try {
      const res = await API.get('/experiments');
      const data = res.data;
      setExperiments(Array.isArray(data) ? data : data?.experiments || []);
    } catch (e) {
      setExperiments([]);
    } finally { setLoading(false); }
  }, []);

  useEffect(() => { load(); }, [load]);

  return (
    <DashboardLayout title="Experiments">
      <motion.div initial="hidden" animate="visible" variants={{ visible: { transition: { staggerChildren: 0.06 } } }}>
        <motion.div variants={fadeUp} className="mb-8">
          <h1 className="font-display text-2xl font-bold uppercase tracking-tight">Experiments</h1>
          <p className="font-mono text-[11px] text-white/30 tracking-wider uppercase mt-1">{experiments.length} total runs</p>
        </motion.div>

        {loading ? (
          <div className="flex justify-center py-20">
            <div className="w-6 h-6 border-2 border-[#B7FF4A] border-t-transparent rounded-full animate-spin" />
          </div>
        ) : !Array.isArray(experiments) || experiments.length === 0 ? (
          <motion.div variants={fadeUp} className="border border-white/[.06] p-16 text-center">
            <p className="font-mono text-[11px] text-white/25 tracking-wider uppercase">No experiments yet. Train a model to see results here.</p>
          </motion.div>
        ) : (
          <motion.div variants={fadeUp} className="border border-white/[.06]">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-white/[.06]">
                    {['#', 'Best Model', 'Score', 'Type', 'Rows', 'Models', 'Version', 'Time'].map(h => (
                      <th key={h} className="px-5 py-3 text-left font-mono text-[10px] text-white/25 tracking-[0.15em] uppercase font-normal">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {Array.isArray(experiments) && experiments.map((exp, i) => (
                    <tr key={i} className="border-b border-white/[.04] hover:bg-white/[.02] transition-colors">
                      <td className="px-5 py-3 font-mono text-[11px] text-white/25">{experiments.length - i}</td>
                      <td className="px-5 py-3 font-mono text-[12px] text-white/70">{exp.best_model || exp.model_name || '-'}</td>
                      <td className="px-5 py-3 font-mono text-[12px] text-[#B7FF4A]">{safeNumber(exp.score)}</td>
                      <td className="px-5 py-3">
                        <span className={`font-mono text-[10px] tracking-wider uppercase px-2 py-0.5 border ${exp.problem_type === 'classification' ? 'border-[#6AA7FF]/20 text-[#6AA7FF]' : 'border-[#FFCC66]/20 text-[#FFCC66]'}`}>
                          {exp.problem_type || '-'}
                        </span>
                      </td>
                      <td className="px-5 py-3 font-mono text-[11px] text-white/40">{exp.rows || '-'}</td>
                      <td className="px-5 py-3 font-mono text-[11px] text-white/40">{exp.total_models || '-'}</td>
                      <td className="px-5 py-3 font-mono text-[10px] text-white/25">{exp.model_version || '-'}</td>
                      <td className="px-5 py-3 font-mono text-[10px] text-white/20">{exp.time || exp.timestamp || '-'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </motion.div>
        )}
      </motion.div>
    </DashboardLayout>
  );
}
