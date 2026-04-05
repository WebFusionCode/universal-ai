import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import DashboardLayout from '../components/DashboardLayout';
import API from '../lib/api';

const fadeUp = { hidden: { opacity: 0, y: 20 }, visible: { opacity: 1, y: 0 } };

export default function Predict() {
  const [file, setFile] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handlePredict = async () => {
    if (!file) { setError('Upload a file first'); return; }
    setLoading(true); setError(''); setPredictions(null);
    try {
      const fd = new FormData(); fd.append('file', file);
      const res = await API.post('/api/predict', fd);
      if (res.data.error) setError(res.data.error);
      else setPredictions(res.data);
    } catch (err) {
      setError('Prediction failed: ' + (err.response?.data?.detail || err.message));
    } finally { setLoading(false); }
  };

  return (
    <DashboardLayout title="Predict">
      <motion.div initial="hidden" animate="visible" variants={{ visible: { transition: { staggerChildren: 0.06 } } }}>
        <motion.div variants={fadeUp} className="mb-8">
          <h1 className="font-display text-2xl font-bold uppercase tracking-tight">Predictions</h1>
          <p className="font-mono text-[11px] text-white/30 tracking-wider uppercase mt-1">Run predictions on new data</p>
        </motion.div>

        <motion.div variants={fadeUp} className="border border-white/[.06] p-6 mb-6">
          <h3 className="font-mono text-[10px] text-white/30 tracking-[0.15em] uppercase mb-5">Upload Data for Prediction</h3>
          <div className={`border-2 border-dashed p-10 text-center cursor-pointer transition-all duration-300 ${file ? 'border-[#6AA7FF]/30 bg-[#6AA7FF]/[.02]' : 'border-white/[.08] hover:border-white/[.15]'}`}
            onClick={() => document.getElementById('predict-file').click()} data-testid="predict-upload">
            <input id="predict-file" type="file" accept=".csv,.xlsx,.json" className="hidden"
              onChange={(e) => { setFile(e.target.files[0]); setPredictions(null); setError(''); }} data-testid="predict-file-input" />
            {file ? (
              <div><p className="font-display text-base font-bold text-white">{file.name}</p><p className="font-mono text-[10px] text-white/25 tracking-wider uppercase mt-1">Click to replace</p></div>
            ) : (
              <div><p className="font-display text-base text-white/30">Drop prediction data here</p><p className="font-mono text-[10px] text-white/15 tracking-wider uppercase mt-1">CSV / Excel / JSON</p></div>
            )}
          </div>
          <button data-testid="predict-btn" onClick={handlePredict} disabled={!file || loading}
            className="mt-5 w-full py-3 bg-[#6AA7FF] text-white font-mono text-[11px] font-bold tracking-[0.1em] uppercase hover:bg-[#7bb8ff] transition-all disabled:opacity-30">
            {loading ? 'Running predictions...' : 'Run Predictions'}
          </button>
        </motion.div>

        {error && <div className="mb-6 border border-[#FF5C7A]/20 bg-[#FF5C7A]/5 px-6 py-4 font-mono text-[11px] text-[#FF5C7A]">{error}</div>}

        <AnimatePresence>
          {predictions && (
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="border border-white/[.06]">
              <div className="px-6 py-4 border-b border-white/[.06] flex items-center gap-4">
                <h3 className="font-mono text-[10px] text-white/40 tracking-[0.15em] uppercase">Predictions</h3>
                <span className="font-mono text-[10px] text-[#5b9ea6] tracking-wider border border-[#5b9ea6]/20 px-2 py-0.5">{predictions.problem_type}</span>
                <span className="font-mono text-[10px] text-white/25 tracking-wider">{predictions.num_predictions} results</span>
              </div>
              <div className="overflow-x-auto max-h-80">
                <table className="w-full">
                  <thead className="sticky top-0 bg-[#0a0a0a]">
                    <tr className="border-b border-white/[.08]">
                      <th className="px-6 py-3 text-left font-mono text-[10px] text-white/25 tracking-wider uppercase font-normal">#</th>
                      <th className="px-6 py-3 text-left font-mono text-[10px] text-white/25 tracking-wider uppercase font-normal">Prediction</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(predictions.predictions || []).slice(0, 100).map((p, i) => (
                      <tr key={i} className="border-b border-white/[.04]">
                        <td className="px-6 py-2.5 font-mono text-[11px] text-white/25">{i + 1}</td>
                        <td className="px-6 py-2.5 font-mono text-[12px] text-[#B7FF4A]">{typeof p === 'object' ? p.prediction : p}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    </DashboardLayout>
  );
}
