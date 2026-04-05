import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import DashboardLayout from '../components/DashboardLayout';
import API from '../lib/api';

const fadeUp = { hidden: { opacity: 0, y: 20 }, visible: { opacity: 1, y: 0 } };

export default function Train() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [targetColumn, setTargetColumn] = useState('');
  const [training, setTraining] = useState(false);
  const [progress, setProgress] = useState(0);
  const [statusMsg, setStatusMsg] = useState('');
  const [results, setResults] = useState(null);
  const [error, setError] = useState('');
  const [uploading, setUploading] = useState(false);
  const [activeTab, setActiveTab] = useState('results');
  const fileRef = useRef(null);
  const wsRef = useRef(null);

  useEffect(() => {
    if (!training) return;
    const backendUrl = process.env.REACT_APP_BACKEND_URL || '';
    const wsUrl = backendUrl.replace(/^http/, 'ws') + '/api/ws/progress';
    try {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;
      ws.onmessage = (evt) => {
        try {
          const data = JSON.parse(evt.data);
          setProgress(data.progress || 0);
          setStatusMsg(data.message || data.status || '');
        } catch (e) {}
      };
      return () => ws.close();
    } catch (e) {}
  }, [training]);

  const handleFileSelect = async (selectedFile) => {
    setFile(selectedFile);
    setPreview(null);
    setTargetColumn('');
    setResults(null);
    setError('');
    setUploading(true);
    try {
      const fd = new FormData();
      fd.append('file', selectedFile);
      const res = await API.post('/api/preview', fd);
      setPreview(res.data);
      if (res.data.suggested_target_columns?.length > 0) {
        setTargetColumn(res.data.suggested_target_columns[0]);
      }
    } catch (err) {
      const errMsg = err.response?.data?.detail || err.message;
      console.error("Preview Bug Details:", err.response || err);
      setError('Failed to preview: ' + errMsg);
    } finally {
      setUploading(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const f = e.dataTransfer.files[0];
    if (f) handleFileSelect(f);
  };

  const handleTrain = async () => {
    if (!file || !targetColumn) { setError('Upload a file and select target column'); return; }
    setTraining(true);
    setProgress(0);
    setResults(null);
    setError('');
    setStatusMsg('Initializing...');
    try {
      const fd = new FormData();
      fd.append('file', file);
      fd.append('target_column', targetColumn);
      const res = await API.post('/api/train', fd);
      if (res.data.error) {
        setError(res.data.error);
      } else {
        setResults(res.data);
        setProgress(100);
        setStatusMsg('Training completed!');
      }
    } catch (err) {
      setError('Training failed: ' + (err.response?.data?.detail || err.message));
    } finally {
      setTraining(false);
    }
  };

  return (
    <DashboardLayout title="Train Model">
      <motion.div initial="hidden" animate="visible" variants={{ visible: { transition: { staggerChildren: 0.06 } } }}>
        <motion.div variants={fadeUp} className="mb-8">
          <h1 className="font-display text-2xl font-bold uppercase tracking-tight">Train Model</h1>
          <p className="font-mono text-[11px] text-white/30 tracking-wider uppercase mt-1">Upload dataset and train ML models</p>
        </motion.div>

        {/* Step indicator */}
        <motion.div variants={fadeUp} className="flex items-center gap-8 mb-8">
          {[
            { num: '01', label: 'Upload', active: true },
            { num: '02', label: 'Configure', active: !!preview },
            { num: '03', label: 'Train', active: training || !!results },
            { num: '04', label: 'Results', active: !!results },
          ].map((step, i) => (
            <div key={i} className={`flex items-center gap-2 ${step.active ? 'text-white' : 'text-white/20'} transition-colors`}>
              <span className="font-mono text-[10px] tracking-wider">{step.num}</span>
              <span className={`font-mono text-[10px] tracking-[0.12em] uppercase ${step.active ? 'text-[#B7FF4A]' : ''}`}>{step.label}</span>
              {i < 3 && <span className="ml-4 w-8 h-[1px] bg-white/10" />}
            </div>
          ))}
        </motion.div>

        {/* Upload Area */}
        <motion.div variants={fadeUp}
          className={`border-2 border-dashed p-10 text-center cursor-pointer transition-all duration-300 mb-6 ${file ? 'border-[#B7FF4A]/30 bg-[#B7FF4A]/[.02]' : 'border-white/[.08] hover:border-white/[.15]'}`}
          onClick={() => fileRef.current?.click()}
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
          data-testid="upload-area">
          <input ref={fileRef} type="file" accept=".csv,.xlsx,.json" className="hidden"
            onChange={(e) => e.target.files[0] && handleFileSelect(e.target.files[0])} data-testid="file-input" />
          {uploading ? (
            <div>
              <div className="w-6 h-6 border-2 border-[#B7FF4A] border-t-transparent rounded-full animate-spin mx-auto mb-3" />
              <p className="font-mono text-[11px] text-white/40 tracking-wider uppercase">Analyzing dataset...</p>
            </div>
          ) : file ? (
            <div>
              <p className="font-display text-lg font-bold text-white mb-1">{file.name}</p>
              <p className="font-mono text-[10px] text-white/30 tracking-wider uppercase">Click or drag to replace</p>
            </div>
          ) : (
            <div>
              <p className="font-display text-lg font-bold text-white/40 mb-2">Drop your dataset here</p>
              <p className="font-mono text-[10px] text-white/20 tracking-wider uppercase">CSV / Excel / JSON supported</p>
            </div>
          )}
        </motion.div>

        {/* Dataset Preview */}
        <AnimatePresence>
          {preview && (
            <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} exit={{ opacity: 0, height: 0 }} className="mb-6">
              <div className="border border-white/[.06]">
                <div className="px-6 py-4 border-b border-white/[.06] flex items-center gap-4">
                  <h3 className="font-mono text-[10px] tracking-[0.15em] uppercase text-white/40">Dataset Preview</h3>
                  <span className="font-mono text-[10px] tracking-wider text-[#B7FF4A] px-2 py-0.5 border border-[#B7FF4A]/20">{preview.rows} rows</span>
                  <span className="font-mono text-[10px] tracking-wider text-[#6AA7FF] px-2 py-0.5 border border-[#6AA7FF]/20">{preview.columns?.length} columns</span>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b border-white/[.06]">
                        {preview.columns?.map((col, i) => (
                          <th key={i} className="px-4 py-2.5 text-left font-mono text-[10px] text-white/25 tracking-wider uppercase font-normal whitespace-nowrap">{col}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {preview.preview?.map((row, i) => (
                        <tr key={i} className="border-b border-white/[.04]">
                          {preview.columns?.map((col, j) => (
                            <td key={j} className="px-4 py-2 font-mono text-[11px] text-white/50 whitespace-nowrap">
                              {row[col] != null ? String(row[col]).substring(0, 30) : '-'}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Target Selection */}
              <div className="mt-4 border border-white/[.06] p-6">
                <label className="font-mono text-[10px] text-white/30 tracking-[0.15em] uppercase mb-3 block">Select Target Column</label>
                <select data-testid="target-select" value={targetColumn} onChange={(e) => setTargetColumn(e.target.value)}
                  className="w-full px-4 py-3 bg-white/[.03] border border-white/[.08] text-white font-mono text-[12px] focus:outline-none focus:border-[#B7FF4A]/40 appearance-none cursor-pointer">
                  <option value="" className="bg-[#111]">Choose a column...</option>
                  {preview.columns?.map((col, i) => (
                    <option key={i} value={col} className="bg-[#111]">{col}</option>
                  ))}
                </select>
                <button data-testid="train-btn" onClick={handleTrain} disabled={training || !targetColumn}
                  className="mt-4 w-full py-3 bg-[#B7FF4A] text-[#0a0a0a] font-mono text-[11px] font-bold tracking-[0.1em] uppercase hover:bg-[#c8ff73] transition-all disabled:opacity-30 disabled:cursor-not-allowed">
                  {training ? 'Training in progress...' : 'Start Training'}
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Training Progress */}
        <AnimatePresence>
          {(training || progress > 0) && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="mb-6">
              <div className="border border-white/[.06] p-6">
                <div className="flex justify-between items-center mb-4">
                  <span className="font-mono text-[10px] text-white/40 tracking-[0.15em] uppercase">Training Progress</span>
                  <span className="font-mono text-[13px] text-[#B7FF4A] font-bold">{progress}%</span>
                </div>
                <div className="h-1 bg-white/[.06] overflow-hidden mb-4">
                  <motion.div className="h-full bg-gradient-to-r from-[#B7FF4A] to-[#6AA7FF]" animate={{ width: `${progress}%` }} transition={{ duration: 0.5 }} />
                </div>
                <p className="font-mono text-[10px] text-white/30 tracking-wider">{statusMsg}</p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Error */}
        {error && (
          <div className="mb-6 border border-[#FF5C7A]/20 bg-[#FF5C7A]/5 px-6 py-4 font-mono text-[11px] text-[#FF5C7A]">{error}</div>
        )}

        {/* Results */}
        <AnimatePresence>
          {results && (
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
              {/* Result Stats */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
                {[
                  { label: 'Best Model', value: results.best_model, color: '#B7FF4A' },
                  { label: 'Problem Type', value: results.problem_type, color: '#6AA7FF' },
                  { label: results.accuracy != null ? 'Accuracy' : 'R\u00b2 Score', value: (results.accuracy || results.r2 || 0).toFixed(4), color: '#5b9ea6' },
                  { label: 'Version', value: results.model_version, color: '#FFCC66' },
                ].map((stat, i) => (
                  <div key={i} className="border border-white/[.06] p-5 text-center">
                    <p className="font-mono text-[10px] text-white/25 tracking-wider uppercase mb-2">{stat.label}</p>
                    <p className="font-display text-sm font-bold" style={{ color: stat.color }}>{stat.value}</p>
                  </div>
                ))}
              </div>

              {/* Tabs */}
              <div className="flex gap-0 mb-0 border-b border-white/[.06]">
                {['results', 'insights', 'code'].map(tab => (
                  <button key={tab} onClick={() => setActiveTab(tab)}
                    className={`font-mono text-[10px] tracking-[0.12em] uppercase px-6 py-3 transition-all ${activeTab === tab ? 'text-[#B7FF4A] border-b-2 border-[#B7FF4A]' : 'text-white/30 hover:text-white/50'}`}>
                    {tab}
                  </button>
                ))}
              </div>

              {/* Tab Content */}
              <div className="border border-white/[.06] border-t-0 p-6">
                {activeTab === 'results' && results.leaderboard?.length > 0 && (
                  <div>
                    <h4 className="font-mono text-[10px] text-white/30 tracking-[0.15em] uppercase mb-4">Model Leaderboard</h4>
                    <table className="w-full">
                      <thead>
                        <tr className="border-b border-white/[.06]">
                          {['Rank', 'Model', 'Score', 'Time (s)'].map(h => (
                            <th key={h} className="px-4 py-2.5 text-left font-mono text-[10px] text-white/25 tracking-wider uppercase font-normal">{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {results.leaderboard.map((m, i) => (
                          <tr key={i} className={`border-b border-white/[.04] hover:bg-white/[.02] transition-colors ${m.rank === 1 ? 'bg-[#B7FF4A]/[.03]' : ''}`}>
                            <td className="px-4 py-3 font-mono text-[12px]">{m.rank === 1 ? '\ud83e\udd47' : m.rank === 2 ? '\ud83e\udd48' : m.rank === 3 ? '\ud83e\udd49' : m.rank}</td>
                            <td className="px-4 py-3 font-mono text-[12px] text-white/70">{m.model}</td>
                            <td className="px-4 py-3 font-mono text-[12px] text-[#B7FF4A]">{m.score?.toFixed(4)}</td>
                            <td className="px-4 py-3 font-mono text-[11px] text-white/40">{m.time?.toFixed(2)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
                {activeTab === 'insights' && (
                  <div className="space-y-3">
                    <h4 className="font-mono text-[10px] text-white/30 tracking-[0.15em] uppercase mb-4">AI Insights</h4>
                    {(results.ai_insights || []).map((insight, i) => (
                      <div key={i} className="flex items-start gap-3 py-2 border-b border-white/[.04] last:border-0">
                        <span className="font-mono text-[10px] text-white/15">{String(i + 1).padStart(2, '0')}</span>
                        <p className="font-mono text-[12px] text-white/50 leading-relaxed">{insight}</p>
                      </div>
                    ))}
                  </div>
                )}
                {activeTab === 'code' && results.generated_code && (
                  <div>
                    <h4 className="font-mono text-[10px] text-white/30 tracking-[0.15em] uppercase mb-4">Generated Pipeline</h4>
                    <pre className="bg-white/[.02] border border-white/[.06] p-4 overflow-x-auto">
                      <code className="font-mono text-[11px] text-white/60 leading-relaxed whitespace-pre">{results.generated_code}</code>
                    </pre>
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
