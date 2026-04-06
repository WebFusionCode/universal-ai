import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import DashboardLayout from '../components/DashboardLayout';
import API from '../lib/api';

const fadeUp = { hidden: { opacity: 0, y: 20 }, visible: { opacity: 1, y: 0 } };

export default function Train() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState([]);
  const [columns, setColumns] = useState([]);
  const [targetOptions, setTargetOptions] = useState([]);
  const [target, setTarget] = useState('');
  const [modelType, setModelType] = useState('Auto');
  const [training, setTraining] = useState(false);
  const [progress, setProgress] = useState(0);
  const [statusMsg, setStatusMsg] = useState('');
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [uploading, setUploading] = useState(false);
  const [activeTab, setActiveTab] = useState('results');
  const fileRef = useRef(null);
  const wsRef = useRef(null);

  useEffect(() => {
    if (!training) return;
    try {
      const ws = new WebSocket("wss://automl-x.onrender.com/ws/progress");
      wsRef.current = ws;
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          setProgress(data.progress || 0);
          setStatusMsg(data.message || data.status || '');
        } catch (e) {}
      };
      return () => ws.close();
    } catch (e) {}
  }, [training]);

  const handleDrop = async (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (!file) return;

    await handleFileUpload(file);
  };

  const handleFileUpload = async (file) => {
    setSelectedFile(file);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await API.post("/preview", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      console.log("PREVIEW RESPONSE:", res.data);

      setPreview(res.data.preview || []);
      setColumns(res.data.columns || []);
      setTargetOptions(res.data.suggested_target_columns || []);

    } catch (err) {
      console.error("Preview error:", err);
    }
  };

  const handleTrain = async () => {
    if (!selectedFile || !target) {
      alert("Select file & target column");
      return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("target_column", target);

    try {
      const res = await API.post("/train", formData);

      console.log("TRAIN RESPONSE:", res.data);

      setResult(res.data);
    } catch (err) {
      console.error("Train error:", err);
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
            { num: '03', label: 'Train', active: training || !!result },
            { num: '04', label: 'Results', active: !!result },
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
          className={`border-2 border-dashed p-10 text-center cursor-pointer transition-all duration-300 mb-6 ${selectedFile ? 'border-[#B7FF4A]/30 bg-[#B7FF4A]/[.02]' : 'border-white/[.08] hover:border-white/[.15]'}`}
          onClick={() => fileRef.current?.click()}
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
          data-testid="upload-area">
          <input ref={fileRef} type="file" accept=".csv,.xlsx,.zip" className="hidden"
            onChange={(e) => handleFileUpload(e.target.files[0])} data-testid="file-input" />
          {uploading ? (
            <div>
              <div className="w-6 h-6 border-2 border-[#B7FF4A] border-t-transparent rounded-full animate-spin mx-auto mb-3" />
              <p className="font-mono text-[11px] text-white/40 tracking-wider uppercase">Analyzing dataset...</p>
            </div>
          ) : selectedFile ? (
            <div>
              <p className="font-display text-lg font-bold text-white mb-1">{selectedFile.name}</p>
              <p className="font-mono text-[10px] text-white/30 tracking-wider uppercase">Click or drag to replace</p>
            </div>
          ) : (
            <div>
              <p className="font-display text-lg font-bold text-white/40 mb-2">Drop your dataset here</p>
              <p className="font-mono text-[10px] text-white/20 tracking-wider uppercase">CSV / Excel / JSON supported</p>
            </div>
          )}
        </motion.div>

        {preview.length > 0 && (
          <div className="mt-6">
            <h3 className="text-white mb-3">Dataset Preview</h3>

            <div className="overflow-auto border border-green-500/20 rounded">
              <table className="w-full text-sm">
                <thead>
                  <tr>
                    {columns.map((col) => (
                      <th key={col} className="text-gray-400 px-2 py-1 text-left">
                        {col}
                      </th>
                    ))}
                  </tr>
                </thead>

                <tbody>
                  {preview.map((row, i) => (
                    <tr key={i}>
                      {columns.map((col) => (
                        <td key={col} className="px-2 py-1 text-gray-300">
                          {row[col]}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {preview.length > 0 && (
          <div className="mt-6">
            <select
              className="bg-black border border-green-500 text-white px-3 py-2 rounded"
              onChange={(e) => setTarget(e.target.value)}
            >
              <option className="bg-black">Select Target Column</option>
              {targetOptions.map((col) => (
                <option key={col} className="bg-black">{col}</option>
              ))}
            </select>

            <button
              onClick={handleTrain}
              className="mt-4 ml-4 bg-green-600 hover:bg-green-700 px-4 py-2 rounded text-white"
            >
              Train Model
            </button>
          </div>
        )}

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
          {result && (
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="mt-8">
              {/* Result Stats */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
                {[
                  { label: 'Best Model', value: result.best_model, color: '#B7FF4A' },
                  { label: 'Problem Type', value: result.problem_type, color: '#6AA7FF' },
                  { label: result.accuracy != null ? 'Accuracy' : 'R\u00b2 Score', value: (result.accuracy || result.r2 || 0).toFixed(4), color: '#5b9ea6' },
                  { label: 'Version', value: result.model_version, color: '#FFCC66' },
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
                {activeTab === 'results' && result.leaderboard?.length > 0 && (
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
                        {result.leaderboard.map((m, i) => (
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
                    {(result.ai_insights || []).map((insight, i) => (
                      <div key={i} className="flex items-start gap-3 py-2 border-b border-white/[.04] last:border-0">
                        <span className="font-mono text-[10px] text-white/15">{String(i + 1).padStart(2, '0')}</span>
                        <p className="font-mono text-[12px] text-white/50 leading-relaxed">{insight}</p>
                      </div>
                    ))}
                  </div>
                )}
                {activeTab === 'code' && result.generated_code && (
                  <div>
                    <h4 className="font-mono text-[10px] text-white/30 tracking-[0.15em] uppercase mb-4">Generated Pipeline</h4>
                    <pre className="bg-white/[.02] border border-white/[.06] p-4 overflow-x-auto">
                      <code className="font-mono text-[11px] text-white/60 leading-relaxed whitespace-pre">{result.generated_code}</code>
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
