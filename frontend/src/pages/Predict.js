import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import DashboardLayout from '../components/DashboardLayout';
import API from '../lib/api';

const fadeUp = { hidden: { opacity: 0, y: 20 }, visible: { opacity: 1, y: 0 } };

export default function Predict() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handlePredict = async () => {
    if (!file) { setError('Upload a file first'); return; }
    setLoading(true); setError(''); setResult(null);
    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await API.post("/predict", formData);

      console.log("PREDICT:", res.data);

      setResult(res.data);
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
              onChange={(e) => { setFile(e.target.files[0]); setResult(null); setError(''); }} data-testid="predict-file-input" />
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
          {result && (
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="border border-white/[.06]">
              <div className="px-6 py-4 border-b border-white/[.06] flex items-center gap-4">
                <h3 className="font-mono text-[10px] text-white/40 tracking-[0.15em] uppercase">Predictions</h3>
                {result.problem_type && <span className="font-mono text-[10px] text-[#5b9ea6] tracking-wider border border-[#5b9ea6]/20 px-2 py-0.5">{result.problem_type}</span>}
                {result.predicted_class && <span className="font-mono text-[10px] text-[#FFCC66] tracking-wider border border-[#FFCC66]/20 px-2 py-0.5">Image Inference</span>}
                <span className="font-mono text-[10px] text-white/25 tracking-wider">{result.predictions?.length || 1} results</span>
              </div>
              <div className="p-6 text-white font-mono text-[12px]">
                {result?.problem_type === "Classification" && (
                  <div>
                    {result.predictions.map((p, i) => (
                      <div key={i}>
                        Prediction: {p.prediction} | Confidence: {p.confidence}
                      </div>
                    ))}
                  </div>
                )}

                {result?.problem_type === "Regression" && (
                  <div>
                    {result.predictions.map((p, i) => (
                      <div key={i}>{p}</div>
                    ))}
                  </div>
                )}

                {result?.problem_type?.includes("Time") && (
                  <pre>{JSON.stringify(result.forecast, null, 2)}</pre>
                )}

                {result?.predicted_class && (
                  <div>
                    Image Prediction: {result.predicted_class}
                  </div>
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    </DashboardLayout>
  );
}
