import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import DashboardLayout from '../components/DashboardLayout';
import {
  Upload, FileText, Target, Zap, CheckCircle2, AlertCircle,
  Download, Trophy, Brain, BarChart3, Clock
} from 'lucide-react';
import API from '../lib/api';

const safeNumber = (val, decimals = 4) =>
  (val !== null && val !== undefined && typeof val === 'number') ? val.toFixed(decimals) : 'N/A';

const API_BASE = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000';

export default function Train() {
  const navigate = useNavigate();
  const wsRef = useRef(null);

  const [step, setStep] = useState(1);
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [targetColumn, setTargetColumn] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);
  const [progress, setProgress] = useState(0);
  const [progressStatus, setProgressStatus] = useState('');
  const [currentModel, setCurrentModel] = useState('');
  const [epochInfo, setEpochInfo] = useState('');

  const [selectedModel, setSelectedModel] = useState('auto');
  const [datasetMode, setDatasetMode] = useState('auto'); // NEW: Mode selector

  const [isTraining, setIsTraining] = useState(false);
  const [hpParams, setHpParams] = useState({
    epochs: 3,
    batch_size: 32,
    lr: 0.001,
    hidden: 64,
    layers: 2,
    n_estimators: 100
  });

  useEffect(() => {
    const tabularModels = ['rf', 'xgb', 'catboost', 'lr', 'svm', 'knn', 'dt'];
    const timeSeriesModels = ['lstm', 'gru', 'prophet'];

    if (datasetMode === 'tabular' && timeSeriesModels.includes(selectedModel)) {
      setSelectedModel('auto');
    }

    if (datasetMode === 'time_series' && tabularModels.includes(selectedModel)) {
      setSelectedModel('auto');
    }
  }, [datasetMode, selectedModel]);

  // Connect to WebSocket for live progress
  useEffect(() => {
    if (loading) {
      try {
        const ws = new WebSocket(`${API_BASE.replace('http', 'ws')}/ws/progress`);
        ws.onmessage = (e) => {
          const data = JSON.parse(e.data);
          
          if (data.progress !== undefined) setProgress(data.progress);
          if (data.model) setCurrentModel(data.model);
          if (data.status) setProgressStatus(data.status);
          
          if (data.epoch && data.total_epochs) {
            setEpochInfo(`Epoch ${data.epoch}/${data.total_epochs}`);
          } else if (data.status === 'starting') {
            setEpochInfo('Initializing...');
          } else if (data.status === 'complete') {
            setEpochInfo('Finalized');
          }
          
          if (data.loss !== undefined || data.accuracy !== undefined || data.r2_score !== undefined) {
             const loss = typeof data.loss === 'number' ? `L: ${data.loss.toFixed(4)}` : '';
             const acc = typeof data.accuracy === 'number' ? `A: ${data.accuracy.toFixed(4)}` : '';
             const r2 = typeof data.r2_score === 'number' ? `R²: ${data.r2_score.toFixed(4)}` : '';
             setProgressStatus(`${data.status || 'Training'} | ${loss} ${acc} ${r2}`);
          }
        };
        wsRef.current = ws;
      } catch (_) {}
    } else {
      wsRef.current?.close();
    }
    return () => wsRef.current?.close();
  }, [loading]);

  const handleFileSelect = async (e) => {
    const selectedFile = e.target.files[0];
    if (!selectedFile) return;
    const isZip = selectedFile.name.endsWith('.zip');
    setFile(selectedFile);
    setError('');
    setLoading(true);
    setProgress(5);
    setProgressStatus('Reading file...');

    try {
      if (isZip) {
        setPreview({ columns: [], preview: [], is_image: true });
        setStep(2);
      } else {
        const formData = new FormData();
        formData.append('file', selectedFile);
        const res = await API.post('/preview', formData, {
          headers: { 'Content-Type': 'multipart/form-data' }
        });
        setPreview(res.data || { columns: [], preview: [] });
        if (res.data?.suggested_target_columns?.length > 0) {
          setTargetColumn(res.data.suggested_target_columns[0]);
        }
        setStep(2);
      }
    } catch (err) {
      setError(err.response?.data?.detail || err.response?.data?.error || 'Failed to preview dataset');
    } finally {
      setLoading(false);
      setProgress(0);
    }
  };

  const handleTrain = async () => {
    const isZip = file?.name.endsWith('.zip');
    if (!isZip && !targetColumn) {
      setError('Please select a target column');
      return;
    }
    setLoading(true);
    setIsTraining(true);
    setError('');
    setProgress(10);
    setProgressStatus('Uploading...');

    try {
      const formData = new FormData();
      formData.append('file', file);
      if (!isZip) formData.append('target_column', targetColumn);

      formData.append('dataset_mode', datasetMode); // NEW: Send mode
      formData.append('model_name', selectedModel);
      formData.append('selected_model', selectedModel);
      formData.append('params', JSON.stringify({
        [selectedModel]: hpParams,
        "epochs": hpParams.epochs,
        "batch_size": hpParams.batch_size,
        "lr": hpParams.lr,
        "LSTM": { "hidden": hpParams.hidden, "layers": hpParams.layers, "epochs": hpParams.epochs }
      }));


      const res = await API.post('/train', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      const data = res.data;
      console.log('TRAIN RESPONSE:', data);

      if (data.error) {
        setError(data.error + (data.hint ? ` — ${data.hint}` : ''));
        return;
      }

      setProgress(100);
      setProgressStatus('Complete!');
      setResult(data);
      setStep(3);
    } catch (err) {
      setError(err.response?.data?.detail || err.response?.data?.error || 'Training failed');
    } finally {
      setLoading(false);
      setIsTraining(false);
    }
  };

  const resetForm = () => {
    setStep(1); setFile(null); setPreview(null);
    setTargetColumn(''); setResult(null);
    setError(''); setProgress(0); setProgressStatus('');
    setCurrentModel(''); setEpochInfo('');
    setIsTraining(false);
  };

  return (
    <DashboardLayout>
      <div className="space-y-6">

        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="font-display text-2xl font-bold uppercase tracking-tight text-white">Train Model</h1>
            <p className="font-mono text-[11px] text-white/40 tracking-wider uppercase mt-1">
              Upload • Auto-detect • Train • Evaluate
            </p>
          </div>
        </div>

        {/* Step Indicator */}
        <div className="flex items-center gap-4">
          {[
            { num: 1, label: 'Upload', icon: Upload },
            { num: 2, label: 'Configure', icon: Target },
            { num: 3, label: 'Results', icon: CheckCircle2 }
          ].map(({ num, label, icon: Icon }) => (
            <div key={num} className="flex items-center gap-2">
              <div className={`flex items-center justify-center w-8 h-8 rounded-full border ${
                step >= num ? 'bg-[#B7FF4A] border-[#B7FF4A] text-[#0a0a0a]' : 'bg-transparent border-white/20 text-white/40'
              }`}>
                <span className="font-mono text-xs font-bold">{num}</span>
              </div>
              <span className={`font-mono text-[10px] uppercase tracking-wider ${step >= num ? 'text-white' : 'text-white/40'}`}>{label}</span>
              {num < 3 && <div className={`h-[1px] w-8 ${step > num ? 'bg-[#B7FF4A]' : 'bg-white/20'}`} />}
            </div>
          ))}
        </div>

        {/* Progress Bar */}
        {loading && (
          <div className="border border-white/[.08] bg-[#0a0a0a] p-6 space-y-4">
            <div className="flex justify-between items-end">
              <div>
                <div className="font-mono text-[9px] text-[#B7FF4A] uppercase tracking-[0.2em] mb-1 leading-none">
                   Active Engine
                </div>
                <div className="font-display text-lg font-bold text-white uppercase tracking-tight">
                  {currentModel || 'Neural Processor'}
                </div>
              </div>
              <div className="text-right">
                <div className="font-mono text-[10px] text-white/40 uppercase mb-1">
                   {epochInfo || 'In Progress'}
                </div>
                <div className="font-mono text-xl text-[#B7FF4A] font-bold">
                  {progress}%
                </div>
              </div>
            </div>

            <div className="w-full h-2 bg-white/[.04] rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-[#B7FF4A]/40 to-[#B7FF4A] rounded-full transition-all duration-700 ease-out shadow-[0_0_15px_rgba(183,255,74,0.4)]"
                style={{ width: `${progress}%` }}
              />
            </div>

            <div className="flex items-center gap-3">
               <div className="w-1.5 h-1.5 rounded-full bg-[#B7FF4A] animate-ping" />
               <span className="font-mono text-[10px] text-white/40 uppercase tracking-widest leading-none">
                 {progressStatus || 'Synchronizing Neural Paths...'}
               </span>
            </div>
            {isTraining && (
              <div className="mt-4 flex justify-center">
                <button
                  onClick={() => navigate('/model-explain?live=true')}
                  className="px-6 py-2 bg-[#B7FF4A] text-black font-bold uppercase tracking-[0.2em] animate-pulse rounded-lg"
                >
                  ⚡ LIVE MODEL EXPLAIN
                </button>
              </div>
            )}
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="border border-[#FF5C7A]/20 bg-[#FF5C7A]/5 px-4 py-3 flex items-start gap-3">
            <AlertCircle size={16} className="text-[#FF5C7A] mt-0.5 shrink-0" />
            <span className="font-mono text-[11px] text-[#FF5C7A]">{error}</span>
          </div>
        )}

        {/* Step 1: Upload */}
        {step === 1 && (
          <div className="border border-white/[.08] bg-[#111] p-8">
            <div className="text-center">
              <Upload size={48} className="text-[#B7FF4A] mx-auto mb-4" />
              <h3 className="font-display text-lg font-bold uppercase text-white mb-2">Upload Dataset</h3>
              <p className="font-mono text-[11px] text-white/40 mb-6">
                CSV · Excel · ZIP (images) — type auto-detected
              </p>
              <label className="cursor-pointer inline-block">
                <div className="px-6 py-3 bg-[#B7FF4A] text-[#0a0a0a] font-mono text-[11px] font-bold tracking-[0.1em] uppercase hover:bg-[#c8ff73] transition-all">
                  Select File
                </div>
                <input 
                  type="file" 
                  accept="image/*,.csv,.xlsx,.xls,.zip" 
                  onChange={handleFileSelect} 
                  className="hidden" 
                  disabled={loading} 
                />
              </label>
            </div>
          </div>
        )}

        {/* Step 2: Configure */}
        {step === 2 && preview && (
          <div className="space-y-6">
            {/* Stats */}
            <div className="border border-white/[.08] bg-[#111] p-6">
              <h3 className="font-display text-sm font-bold uppercase text-white mb-4 flex items-center gap-2">
                <FileText size={16} className="text-[#B7FF4A]" /> Dataset Preview
              </h3>
              <div className="grid grid-cols-3 gap-4 mb-6">
                <div>
                  <div className="font-mono text-[10px] text-white/40 uppercase mb-1">Rows</div>
                  <div className="font-mono text-xl text-white font-bold">{preview.rows || 0}</div>
                </div>
                <div>
                  <div className="font-mono text-[10px] text-white/40 uppercase mb-1">Columns</div>
                  <div className="font-mono text-xl text-white font-bold">{preview.columns?.length || 0}</div>
                </div>
                <div>
                  <div className="font-mono text-[10px] text-white/40 uppercase mb-1">Detected Type</div>
                  <div className="font-mono text-xl text-[#B7FF4A] font-bold capitalize">{preview.dataset_type || 'Auto'}</div>
                </div>
              </div>

              {/* Preview Table */}
              {preview.preview?.length > 0 && (
                <div className="overflow-x-auto">
                  <table className="w-full border border-white/[.08]">
                    <thead>
                      <tr className="bg-white/[.03]">
                        {Object.keys(preview.preview[0]).map((col) => (
                          <th key={col} className="px-4 py-2 text-left font-mono text-[10px] text-white/60 uppercase">{col}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {preview.preview.map((row, idx) => (
                        <tr key={idx} className="border-t border-white/[.08]">
                          {Object.values(row).map((val, i) => (
                            <td key={i} className="px-4 py-2 font-mono text-[11px] text-white/80">{String(val ?? '')}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>

            {/* Target Selection */}
            <div className="border border-white/[.08] bg-[#111] p-6">
              <h3 className="font-display text-sm font-bold uppercase text-white mb-4 flex items-center gap-2">
                <Target size={16} className="text-[#B7FF4A]" /> Target Column
              </h3>

              {!preview?.is_image ? (
                <div className="space-y-3">
                  <select
                    value={targetColumn}
                    onChange={(e) => setTargetColumn(e.target.value)}
                    className="w-full px-4 py-3 bg-white/[.03] border border-white/[.08] text-white font-mono text-[13px] focus:outline-none focus:border-[#B7FF4A]/40 transition-all"
                    data-testid="target-column-select"
                  >
                    <option value="">Select target column...</option>
                    {Array.isArray(preview?.columns) && preview.columns.map((col) => (
                      <option key={col} value={col}>{col}</option>
                    ))}
                  </select>
                  {targetColumn && (
                    <p className="font-mono text-[10px] text-[#B7FF4A]">✓ Target: <strong>{targetColumn}</strong></p>
                  )}
                </div>
              ) : (
                <div className="px-4 py-3 bg-white/[.03] border border-white/[.08] text-white font-mono text-[13px]">
                  📁 Image dataset detected — Neural network tournament will run
                </div>
              )}

              {/* Model Selection */}
              <div className="mt-6 flex flex-col md:flex-row gap-6">

                <div className="flex-1 space-y-3">
                  <h3 className="font-display text-sm font-bold uppercase text-white mb-4 flex items-center gap-2">
                    <Zap size={16} className="text-[#B7FF4A]" /> Dataset Mode
                  </h3>
                  <select
                    value={datasetMode}
                    onChange={(e) => setDatasetMode(e.target.value)}
                    className="w-full px-4 py-3 bg-white/[.03] border border-white/[.08] text-white font-mono text-[13px] focus:outline-none focus:border-[#B7FF4A]/40 transition-all"
                  >
                    <option value="auto">Auto (Recommended)</option>
                    <option value="tabular">📊 Tabular (RF/XGB)</option>
                    <option value="time_series">📈 Time Series (Prophet/LSTM)</option>
                  </select>
                  <p className="font-mono text-[10px] text-white/60">
                    {datasetMode === 'tabular' ? '✅ RandomForest, XGBoost, CatBoost will compete' :
                     datasetMode === 'time_series' ? '⏰ Prophet, LSTM/GRU forecasting pipeline' :
                     '🤖 AI will auto-select best approach'}
                  </p>
                </div>

                <div className="flex-1 space-y-3">
                  <h3 className="font-display text-sm font-bold uppercase text-white mb-4 flex items-center gap-2">
                    <Brain size={16} className="text-[#B7FF4A]" /> Model Selection
                  </h3>
                  <select
                    value={selectedModel}
                    onChange={(e) => setSelectedModel(e.target.value)}
                    className="w-full px-4 py-3 bg-white/[.03] border border-white/[.08] text-white font-mono text-[13px] focus:outline-none focus:border-[#B7FF4A]/40 transition-all"
                  >
                    <option value="auto">Auto (Compare All Models)</option>
                    
                    {/* TABULAR */}
                    <optgroup label="Tabular Intelligence">
                      <option value="rf">Random Forest</option>
                      <option value="xgb">XGBoost</option>
                      <option value="catboost">CatBoost</option>
                      <option value="lr">Linear/Logistic Regression</option>
                      <option value="svm">SVM / SVR</option>
                      <option value="knn">KNN</option>
                      <option value="dt">Decision Tree</option>
                    </optgroup>

                    {/* TIME SERIES */}
                    <optgroup label="Time Series Forecasting">
                      <option value="lstm">LSTM (Deep Learning)</option>
                      <option value="gru">GRU (RNN)</option>
                      <option value="prophet">Prophet (Meta)</option>
                    </optgroup>

                    {/* DEEP LEARNING / VISION */}
                    <optgroup label="Advanced Architectures (Vision)">
                      <option value="cnn">Simple CNN</option>
                      <option value="mobilenet">MobileNet</option>
                      <option value="resnet">ResNet</option>
                      <option value="efficientnet">EfficientNet</option>
                      <option value="vit">Vision Transformer</option>
                      <option value="unet">UNet (Segmentation)</option>
                    </optgroup>
                  </select>
                </div>


                <div className="flex-1 space-y-3">
                  <h3 className="font-display text-sm font-bold uppercase text-white mb-4 flex items-center gap-2">
                    <Zap size={16} className="text-[#B7FF4A]" /> Hyperparameters
                  </h3>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-1">
                      <span className="font-mono text-[9px] text-white/20 uppercase">Epochs / Estimators</span>
                      <input 
                        type="number"
                        value={hpParams.epochs}
                        onChange={(e) => setHpParams({...hpParams, epochs: parseInt(e.target.value) || 1, n_estimators: parseInt(e.target.value) || 100})}
                        className="w-full bg-white/[.02] border border-white/[.08] rounded px-3 py-2 text-white font-mono text-xs"
                      />
                    </div>
                    <div className="space-y-1">
                      <span className="font-mono text-[9px] text-white/20 uppercase">Batch Size</span>
                      <input 
                        type="number"
                        value={hpParams.batch_size}
                        onChange={(e) => setHpParams({...hpParams, batch_size: parseInt(e.target.value) || 32})}
                        className="w-full bg-white/[.02] border border-white/[.08] rounded px-3 py-2 text-white font-mono text-xs"
                      />
                    </div>
                  </div>
                </div>
              </div>

              <div className="flex gap-3 mt-6">
                <button
                  onClick={handleTrain}
                  disabled={!file || loading}
                  className="px-6 py-3 bg-[#B7FF4A] text-[#0a0a0a] font-mono text-[11px] font-bold tracking-[0.1em] uppercase hover:bg-[#c8ff73] transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                  data-testid="train-button"
                >
                  <Zap size={14} />
                  {loading ? 'Training...' : 'Train Model'}
                </button>
                <button
                  onClick={resetForm}
                  disabled={loading}
                  className="px-6 py-3 border border-white/[.08] text-white font-mono text-[11px] font-bold tracking-[0.1em] uppercase hover:bg-white/[.03] transition-all disabled:opacity-50"
                >
                  Reset
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Step 3: Results */}
        {step === 3 && result && (
          <div className="space-y-6">
            {/* Success Banner */}
            <div className="border border-[#B7FF4A]/20 bg-[#B7FF4A]/5 px-6 py-4 flex items-center gap-3">
              <CheckCircle2 size={20} className="text-[#B7FF4A]" />
              <div>
                <p className="font-mono text-[12px] text-[#B7FF4A] font-bold">Training Complete!</p>
                <p className="font-mono text-[10px] text-white/40 mt-0.5 uppercase">
                  {result.problem_type} · {result.dataset_type} · {result.rows ?? '-'} rows
                </p>
              </div>
            </div>

            {/* Metrics Grid */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {[
                { label: 'Best Model', value: result.best_model || 'N/A', icon: Trophy, color: '#B7FF4A' },
                { label: 'Score', value: safeNumber(result.score), icon: BarChart3, color: '#6AA7FF' },
                { label: 'Loss', value: safeNumber(result.loss), icon: Zap, color: '#FF6B6B' },
                { label: 'Problem Type', value: result.problem_type || 'N/A', icon: Brain, color: '#FF6B9D' },
                { label: 'Dataset Rows', value: result.rows ?? 'N/A', icon: Clock, color: '#4FD1C5' }
              ].map(({ label, value, icon: Icon, color }) => (
                <div key={label} className="border border-white/[.08] bg-[#111] p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <Icon size={13} style={{ color }} />
                    <div className="font-mono text-[10px] text-white/40 uppercase">{label}</div>
                  </div>
                  <div className="font-mono text-lg font-bold" style={{ color }}>{value}</div>
                </div>
              ))}
            </div>

            {/* Extra metrics (MAE, accuracy, R2, etc.) */}
            {result.metrics && Object.keys(result.metrics).length > 0 && (
              <div className="border border-white/[.08] bg-[#111] p-5">
                <h3 className="font-display text-sm font-bold uppercase text-white mb-4 flex items-center gap-2">
                  <BarChart3 size={14} className="text-[#B7FF4A]" /> Detailed Metrics
                </h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  {Object.entries(result.metrics).map(([key, val]) => (
                    <div key={key} className="bg-white/[.02] border border-white/[.06] p-3">
                      <div className="font-mono text-[10px] text-white/40 uppercase mb-1">{key.replace(/_/g, ' ')}</div>
                      <div className="font-mono text-base text-[#B7FF4A] font-bold">{safeNumber(val)}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Leaderboard */}
            {Array.isArray(result.leaderboard) && result.leaderboard.length > 0 && (
              <div className="border border-white/[.08] bg-[#111] p-5">
                <h3 className="font-display text-sm font-bold uppercase text-white mb-4 flex items-center gap-2">
                  <Trophy size={14} className="text-[#B7FF4A]" /> Model Leaderboard
                </h3>
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b border-white/[.08]">
                        {['Rank', 'Model', 'Score', 'Loss', 'Time (s)'].map(h => (
                          <th key={h} className="px-4 py-2 text-left font-mono text-[10px] text-white/40 uppercase font-normal">{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {result.leaderboard.map((m, i) => (
                        <tr
                          key={i}
                          className={`border-b border-white/[.04] transition-colors ${i === 0 ? 'bg-[#B7FF4A]/[.04]' : 'hover:bg-white/[.02]'}`}
                        >
                          <td className="px-4 py-2.5 font-mono text-[11px] text-white/40">
                            {i === 0 ? '🥇' : i === 1 ? '🥈' : i === 2 ? '🥉' : `#${i + 1}`}
                          </td>
                          <td className={`px-4 py-2.5 font-mono text-[12px] font-bold ${i === 0 ? 'text-[#B7FF4A]' : 'text-white/80'}`}>
                            {m.model}
                          </td>
                          <td className="px-6 py-4 font-mono text-[12px] text-[#B7FF4A]">
                            {safeNumber(m.score)}
                          </td>
                           <td className="px-6 py-4 font-mono text-[12px] text-[#FF6B6B]">
                            {safeNumber(m.loss)}
                          </td>
                          <td className="px-4 py-2.5 font-mono text-[11px] text-white/30">
                            {m.time !== null && m.time !== undefined ? safeNumber(m.time, 2) : '—'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                {/* Vertical Model Comparisons */}
                <div className="mt-8 space-y-4">
                  <h4 className="font-mono text-[10px] text-white/30 uppercase tracking-[0.2em] mb-4">Detailed Comparisons</h4>
                  {result.all_models?.map((m) => (
                    <div key={m.model} className="space-y-1.5">
                      <div className="flex justify-between items-end">
                        <span className="font-mono text-[11px] text-white/80">{m.model}</span>
                        <span className="font-mono text-[11px] text-[#B7FF4A]">{safeNumber(m.score)}</span>
                      </div>
                      <div className="w-full h-1 bg-white/[.04] rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-white/20 rounded-full"
                          style={{ width: `${Math.min(100, (m.score > 0 ? m.score : 0) * 100)}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* AI Insights */}
            {Array.isArray(result.ai_insights) && result.ai_insights.length > 0 && (
              <div className="border border-white/[.08] bg-[#111] p-5">
                <h3 className="font-display text-sm font-bold uppercase text-white mb-4 flex items-center gap-2">
                  <Brain size={14} className="text-[#B7FF4A]" /> AI Insights
                </h3>
                <div className="space-y-2">
                  {result.ai_insights.map((ins, i) => (
                    <div key={i} className="flex items-start gap-3 py-2 border-b border-white/[.04] last:border-0">
                      <span className="text-[#B7FF4A] font-bold shrink-0">→</span>
                      <p className="font-mono text-[11px] text-white/60">{ins}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Download + Actions */}
            <div className="border border-white/[.08] bg-[#111] p-5">
              <h3 className="font-display text-sm font-bold uppercase text-white mb-4 flex items-center gap-2">
                <Download size={14} className="text-[#B7FF4A]" /> Export
              </h3>
              <div className="flex flex-wrap gap-3">
                <a
                  href={`${API_BASE}/download-code/python`}
                  target="_blank"
                  rel="noreferrer"
                  className="px-4 py-2 border border-white/[.08] text-white font-mono text-[10px] tracking-wider uppercase hover:border-[#B7FF4A]/40 hover:text-[#B7FF4A] transition-all flex items-center gap-2"
                >
                  <Download size={12} /> Python Script
                </a>
                <a
                  href={`${API_BASE}/download-code/notebook`}
                  target="_blank"
                  rel="noreferrer"
                  className="px-4 py-2 border border-white/[.08] text-white font-mono text-[10px] tracking-wider uppercase hover:border-[#B7FF4A]/40 hover:text-[#B7FF4A] transition-all flex items-center gap-2"
                >
                  <Download size={12} /> Notebook
                </a>
                <a
                  href={`${API_BASE}/download-model`}
                  target="_blank"
                  rel="noreferrer"
                  className="px-4 py-2 bg-[#B7FF4A]/10 border border-[#B7FF4A]/20 text-[#B7FF4A] font-mono text-[10px] tracking-wider uppercase hover:bg-[#B7FF4A]/20 transition-all flex items-center gap-2"
                >
                  <Download size={12} /> Model (.pkl)
                </a>
                <a
                  href={`${API_BASE}/download-code/docker`}
                  target="_blank"
                  rel="noreferrer"
                  className="px-4 py-2 border border-white/[.08] text-white font-mono text-[10px] tracking-wider uppercase hover:border-[#B7FF4A]/40 hover:text-[#B7FF4A] transition-all flex items-center gap-2"
                >
                  <Download size={12} /> Docker Package
                </a>
              </div>
            </div>

            {/* Navigation */}
            <div className="flex gap-3">
              <button
                onClick={() => navigate('/experiments')}
                className="px-6 py-3 bg-[#B7FF4A] text-[#0a0a0a] font-mono text-[11px] font-bold tracking-[0.1em] uppercase hover:bg-[#c8ff73] transition-all"
              >
                View Experiments
              </button>
              <button
                onClick={() => navigate('/predict')}
                className="px-6 py-3 border border-[#6AA7FF]/30 text-[#6AA7FF] font-mono text-[11px] font-bold tracking-[0.1em] uppercase hover:bg-[#6AA7FF]/10 transition-all"
              >
                Run Predictions
              </button>
              <button
                onClick={resetForm}
                className="px-6 py-3 border border-white/[.08] text-white font-mono text-[11px] font-bold tracking-[0.1em] uppercase hover:bg-white/[.03] transition-all"
              >
                Train Another
              </button>
            </div>
          </div>
        )}
      </div>
    </DashboardLayout>
  );
}
