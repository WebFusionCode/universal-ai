import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import DashboardLayout from '../components/DashboardLayout';
import API from '../lib/api';

const fadeUp = { hidden: { opacity: 0, y: 20 }, visible: { opacity: 1, y: 0 } };

export default function Download() {
  const [report, setReport] = useState(null);
  const [explain, setExplain] = useState(null);
  const backendUrl = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000';
  const featureImportanceEntries = Array.isArray(explain?.feature_importance)
    ? explain.feature_importance.map((item) => [item.feature, item.importance])
    : Object.entries(explain?.feature_importance || explain?.coefficients || {});

  useEffect(() => {
    const load = async () => {
      const [r, e] = await Promise.all([
        API.get('/training-report').catch(() => ({ data: {} })),
        API.get('/model-explain').catch(() => ({ data: {} })),
      ]);
      if (!r.data.error) setReport(r.data);
      if (!e.data.error) setExplain(e.data);
    };
    load();
  }, []);

  return (
    <DashboardLayout title="Downloads">
      <motion.div initial="hidden" animate="visible" variants={{ visible: { transition: { staggerChildren: 0.06 } } }}>
        <motion.div variants={fadeUp} className="mb-8">
          <h1 className="font-display text-2xl font-bold uppercase tracking-tight">Download Center</h1>
          <p className="font-mono text-[11px] text-white/30 tracking-wider uppercase mt-1">Export models, code, and reports</p>
        </motion.div>

        {/* Downloads */}
        <motion.div variants={fadeUp} className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-8">
          {[
            { label: 'Neural Project', sub: 'Download Production ZIP', icon: '01', path: '/download-project' },
            { label: 'Best Model', sub: 'Download trained weights', icon: '02', path: '/download-model' },
            { label: 'Documentation', sub: 'Generated Reports', icon: '03', path: '/training-report' },
          ].map((d, i) => (
            <div key={i} className="border border-white/[.06] p-6 cursor-pointer hover:border-white/[.15] hover:bg-white/[.02] transition-all duration-300"
              onClick={() => window.open(`${backendUrl}${d.path}`, '_blank')}>
              <p className="font-mono text-[40px] font-bold text-white/[.04] mb-3">{d.icon}</p>
              <h3 className="font-display text-sm font-bold uppercase tracking-tight text-white mb-1">{d.label}</h3>
              <p className="font-mono text-[10px] text-white/30 tracking-wider">{d.sub}</p>
            </div>
          ))}
        </motion.div>

        {/* Report */}
        {report && (
          <motion.div variants={fadeUp} className="border border-white/[.06] p-6 mb-6">
            <h3 className="font-mono text-[10px] text-white/30 tracking-[0.15em] uppercase mb-5">Training Report</h3>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              <div className="border border-white/[.06] p-5">
                <p className="font-mono text-[10px] text-white/25 tracking-wider uppercase mb-2">Quality</p>
                <p className="font-display text-2xl font-bold text-[#B7FF4A]">{report.dataset_quality?.quality_score}%</p>
                <p className="font-mono text-[10px] text-white/20 mt-1">Missing: {(report.dataset_quality?.missing_ratio * 100)?.toFixed(1)}%</p>
              </div>
              <div className="border border-white/[.06] p-5">
                <p className="font-mono text-[10px] text-white/25 tracking-wider uppercase mb-2">Strength</p>
                <p className="font-display text-2xl font-bold text-[#6AA7FF]">{report.model_strength?.model_strength}</p>
              </div>
              <div className="border border-white/[.06] p-5">
                <p className="font-mono text-[10px] text-white/25 tracking-wider uppercase mb-2">Analysis</p>
                <p className="font-mono text-[11px] text-white/40 leading-relaxed">{report.explanation}</p>
              </div>
            </div>
          </motion.div>
        )}

        {/* Feature Importance */}
        {explain && featureImportanceEntries.length > 0 && (
          <motion.div variants={fadeUp} className="border border-white/[.06] p-6">
            <h3 className="font-mono text-[10px] text-white/30 tracking-[0.15em] uppercase mb-5">Feature Importance</h3>
            <div className="space-y-2">
              {featureImportanceEntries.slice(0, 15).map(([feat, val], i) => {
                const maxVal = Math.max(...featureImportanceEntries.map(([, value]) => Math.abs(Number(value) || 0))) || 1;
                return (
                  <div key={i} className="flex items-center gap-3">
                    <span className="font-mono text-[10px] text-white/35 w-36 truncate" title={feat}>{feat}</span>
                    <div className="flex-1 h-1 bg-white/[.04] overflow-hidden">
                      <motion.div initial={{ width: 0 }} whileInView={{ width: `${Math.min(Math.abs(val) * 100 / maxVal, 100)}%` }}
                        viewport={{ once: true }} transition={{ duration: 0.8, delay: i * 0.05 }}
                        className="h-full bg-gradient-to-r from-[#B7FF4A] to-[#5b9ea6]" />
                    </div>
                    <span className="font-mono text-[10px] text-white/25 w-14 text-right">{Number(val).toFixed(4)}</span>
                  </div>
                );
              })}
            </div>
          </motion.div>
        )}
      </motion.div>
    </DashboardLayout>
  );
}
