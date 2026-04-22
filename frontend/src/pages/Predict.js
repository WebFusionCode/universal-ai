import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import DashboardLayout from '../components/DashboardLayout';
import { Upload, Zap, AlertCircle, CheckCircle2, BarChart3 } from 'lucide-react';
import API from '../lib/api';

const fadeUp = { hidden: { opacity: 0, y: 20 }, visible: { opacity: 1, y: 0 } };

const riskColor = (risk) => {
  if (!risk) return 'text-white/40';
  if (risk.toLowerCase().includes('low')) return 'text-[#B7FF4A]';
  if (risk.toLowerCase().includes('moderate')) return 'text-[#FFCC66]';
  return 'text-[#FF5C7A]';
};

const safeConf = (v) => (typeof v === 'number' ? `${(v * 100).toFixed(1)}%` : '—');

export default function Predict() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);  // image preview URL
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const isImage = (f) => f && (/\.(jpe?g|png|gif|webp)$/i.test(f.name) || f.type.startsWith("image/"));

  const handleFileChange = (f) => {
    setFile(f);
    setPredictions(null);
    setError('');
    // ✅ ISSUE 3: Use file.type for more reliable image detection
    const isImageFile = f && (f.type.startsWith("image/") || /\.(jpe?g|png|gif|webp)$/i.test(f.name));
    if (isImageFile) {
      setPreview(URL.createObjectURL(f));
    } else {
      setPreview(null);
    }
  };

  const handlePredict = async () => {
    if (!file) { setError('Upload a CSV, Excel, or image file first'); return; }
    setLoading(true); setError(''); setPredictions(null);
    try {
      const fd = new FormData(); fd.append('file', file);
      const res = await API.post('/predict', fd, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      if (res.data.error) setError(res.data.error);
      else setPredictions(res.data);
    } catch (err) {
      setError('Prediction failed: ' + (err.response?.data?.detail || err.message));
    } finally { setLoading(false); }
  };

  const isImgResult = predictions?.problem_type === 'Image Classification';
  const isClassification = predictions?.problem_type === 'Classification';
  const preds = predictions?.predictions || [];

  return (
    <DashboardLayout title="Predict">
      <motion.div initial="hidden" animate="visible" variants={{ visible: { transition: { staggerChildren: 0.06 } } }}>

        {/* Header */}
        <motion.div variants={fadeUp} className="mb-8">
          <h1 className="font-display text-2xl font-bold uppercase tracking-tight">Predictions</h1>
          <p className="font-mono text-[11px] text-white/30 tracking-wider uppercase mt-1">
            Upload new data · Get instant predictions
          </p>
        </motion.div>

        {/* Upload */}
        <motion.div variants={fadeUp} className="border border-white/[.06] p-6 mb-6">
          <h3 className="font-mono text-[10px] text-white/30 tracking-[0.15em] uppercase mb-5">Upload Prediction Data</h3>
          <div
            className={`border-2 border-dashed p-10 text-center cursor-pointer transition-all duration-300 ${file ? 'border-[#B7FF4A]/30 bg-[#B7FF4A]/[.02]' : 'border-white/[.08] hover:border-white/[.15]'
              }`}
            onClick={() => document.getElementById('predict-file').click()}
            data-testid="predict-upload"
          >
            <input
              id="predict-file"
              type="file"
              accept=".csv,.xlsx,.jpg,.jpeg,.png,.webp"
              className="hidden"
              onChange={(e) => handleFileChange(e.target.files[0])}
              data-testid="predict-file-input"
            />
            {preview ? (
              <div>
                <img src={preview} alt="preview" className="max-h-40 mx-auto rounded mb-3 object-contain" />
                <p className="font-display text-base font-bold text-white">{file.name}</p>
                <p className="font-mono text-[10px] text-white/25 tracking-wider uppercase mt-1">Click to replace</p>
              </div>
            ) : file ? (
              <div>
                <CheckCircle2 size={28} className="text-[#B7FF4A] mx-auto mb-2" />
                <p className="font-display text-base font-bold text-white">{file.name}</p>
                <p className="font-mono text-[10px] text-white/25 tracking-wider uppercase mt-1">Click to replace</p>
              </div>
            ) : (
              <div>
                <Upload size={28} className="text-white/20 mx-auto mb-2" />
                <p className="font-display text-base text-white/30">Drop prediction data here</p>
                <p className="font-mono text-[10px] text-white/15 tracking-wider uppercase mt-1">CSV · Excel · JPG · PNG</p>
              </div>
            )}
          </div>
          <button
            data-testid="predict-btn"
            onClick={handlePredict}
            disabled={!file || loading}
            className="mt-4 w-full py-3 bg-[#6AA7FF] text-white font-mono text-[11px] font-bold tracking-[0.1em] uppercase hover:bg-[#7bb8ff] transition-all disabled:opacity-30 flex items-center justify-center gap-2"
          >
            <Zap size={14} />
            {loading ? 'Running predictions...' : 'Run Predictions'}
          </button>
        </motion.div>

        {/* Error */}
        {error && (
          <motion.div variants={fadeUp} className="mb-6 border border-[#FF5C7A]/20 bg-[#FF5C7A]/5 px-4 py-3 flex items-center gap-3">
            <AlertCircle size={15} className="text-[#FF5C7A] shrink-0" />
            <span className="font-mono text-[11px] text-[#FF5C7A]">{error}</span>
          </motion.div>
        )}

        {/* Results */}
        <AnimatePresence>
          {predictions && (
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="border border-white/[.06]">

              {/* Header bar */}
              <div className="px-6 py-4 border-b border-white/[.06] flex flex-wrap items-center gap-4">
                <BarChart3 size={14} className="text-[#B7FF4A]" />
                <h3 className="font-mono text-[10px] text-white/40 tracking-[0.15em] uppercase">Results</h3>
                <span className="font-mono text-[10px] text-[#6AA7FF] tracking-wider border border-[#6AA7FF]/20 px-2 py-0.5">
                  {predictions.problem_type}
                </span>
                {!isImgResult && (
                  <span className="font-mono text-[10px] text-white/25 tracking-wider">
                    {predictions.num_predictions} rows
                  </span>
                )}
              </div>

              {/* Image Classification result */}
              {isImgResult && (
                <div className="p-6 space-y-5">
                  {/* Big result card */}
                  <div className="flex flex-col md:flex-row gap-6 items-start">
                    {preview && (
                      <img src={preview} alt="classified" className="max-h-48 rounded object-contain border border-white/[.08]" />
                    )}
                    <div className="flex-1 space-y-4">
                      <div>
                        <p className="font-mono text-[10px] text-white/40 uppercase mb-1">Prediction</p>
                        <p className="font-display text-3xl font-bold text-[#B7FF4A]">{predictions.predicted_class}</p>
                      </div>
                      <div className="flex gap-6">
                        <div>
                          <p className="font-mono text-[10px] text-white/40 uppercase mb-1">Confidence</p>
                          <p className="font-mono text-xl font-bold text-white">{predictions.confidence ? `${(predictions.confidence * 100).toFixed(1)}%` : '—'}</p>
                        </div>
                        <div>
                          <p className="font-mono text-[10px] text-white/40 uppercase mb-1">Risk Level</p>
                          <p className={`font-mono text-xl font-bold ${riskColor(predictions.risk_level)}`}>{predictions.risk_level}</p>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Class probabilities */}
                  {predictions.all_probabilities && (
                    <div>
                      <p className="font-mono text-[10px] text-white/40 uppercase mb-3">All Class Probabilities</p>
                      <div className="space-y-2">
                        {Object.entries(predictions.all_probabilities)
                          .sort(([, a], [, b]) => b - a)
                          .map(([cls, prob]) => (
                            <div key={cls} className="flex items-center gap-3">
                              <span className="font-mono text-[11px] text-white/60 w-28 shrink-0 truncate">{cls}</span>
                              <div className="flex-1 h-2 bg-white/[.06] rounded-full overflow-hidden">
                                <div
                                  className="h-full bg-[#B7FF4A] rounded-full transition-all"
                                  style={{ width: `${(prob * 100).toFixed(1)}%` }}
                                />
                              </div>
                              <span className="font-mono text-[11px] text-[#B7FF4A] w-12 text-right">{(prob * 100).toFixed(1)}%</span>
                            </div>
                          ))}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Forecast (time-series) */}
              {predictions.forecast && (
                <div className="p-6">
                  <p className="font-mono text-[11px] text-white/60 mb-4 uppercase tracking-wider">Forecast (next 10 periods)</p>
                  {Object.entries(predictions.forecast).map(([col, rows]) => (
                    <div key={col} className="mb-6">
                      <p className="font-mono text-[11px] text-[#B7FF4A] mb-2">Column: {col}</p>
                      <div className="overflow-x-auto">
                        <table className="w-full">
                          <thead>
                            <tr className="border-b border-white/[.06]">
                              <th className="px-4 py-2 text-left font-mono text-[10px] text-white/25 uppercase font-normal">#</th>
                              <th className="px-4 py-2 text-left font-mono text-[10px] text-white/25 uppercase font-normal">Date</th>
                              <th className="px-4 py-2 text-left font-mono text-[10px] text-white/25 uppercase font-normal">Forecast</th>
                            </tr>
                          </thead>
                          <tbody>
                            {rows.map((r, i) => (
                              <tr key={i} className="border-b border-white/[.04]">
                                <td className="px-4 py-2 font-mono text-[11px] text-white/25">{i + 1}</td>
                                <td className="px-4 py-2 font-mono text-[11px] text-white/60">{String(r.ds ?? '')}</td>
                                <td className="px-4 py-2 font-mono text-[12px] text-[#B7FF4A] font-bold">{typeof r.yhat === 'number' ? r.yhat.toFixed(4) : r.yhat}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {/* Classification / Regression table */}
              {!predictions.forecast && preds.length > 0 && (
                <div className="overflow-x-auto max-h-96">
                  <table className="w-full">
                    <thead className="sticky top-0 bg-[#0a0a0a]">
                      <tr className="border-b border-white/[.08]">
                        <th className="px-6 py-3 text-left font-mono text-[10px] text-white/25 uppercase font-normal">#</th>
                        <th className="px-6 py-3 text-left font-mono text-[10px] text-white/25 uppercase font-normal">Prediction</th>
                        {isClassification && (
                          <>
                            <th className="px-6 py-3 text-left font-mono text-[10px] text-white/25 uppercase font-normal">Confidence</th>
                            <th className="px-6 py-3 text-left font-mono text-[10px] text-white/25 uppercase font-normal">Risk</th>
                          </>
                        )}
                      </tr>
                    </thead>
                    <tbody>
                      {preds.slice(0, 200).map((p, i) => (
                        <tr key={i} className="border-b border-white/[.04] hover:bg-white/[.02] transition-colors">
                          <td className="px-6 py-2.5 font-mono text-[11px] text-white/25">{i + 1}</td>
                          <td className="px-6 py-2.5 font-mono text-[12px] text-[#B7FF4A] font-bold">
                            {typeof p === 'object' ? p.prediction : p}
                          </td>
                          {isClassification && (
                            <>
                              <td className="px-6 py-2.5 font-mono text-[11px] text-white/60">
                                {safeConf(p?.confidence)}
                              </td>
                              <td className={`px-6 py-2.5 font-mono text-[11px] font-bold ${riskColor(p?.risk_level)}`}>
                                {p?.risk_level ?? '—'}
                              </td>
                            </>
                          )}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {preds.length > 200 && (
                    <p className="px-6 py-3 font-mono text-[10px] text-white/20">Showing first 200 of {preds.length}</p>
                  )}
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    </DashboardLayout>
  );
}
