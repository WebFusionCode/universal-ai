import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line, AreaChart, Area, Cell, PieChart, Pie
} from 'recharts';
import { 
  Brain, BarChart3, Target, Zap, Activity, Image as ImageIcon, 
  Upload, CheckCircle2, AlertCircle, Info, ChevronRight, Search
} from 'lucide-react';
import DashboardLayout from '../components/DashboardLayout';
import API from '../lib/api';

const API_BASE = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000';

const fadeUp = { hidden: { opacity: 0, y: 20 }, visible: { opacity: 1, y: 0 } };

export default function ModelExplain() {
  const [activeTab, setActiveTab] = useState('summary');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [data, setData] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [imageResult, setImageResult] = useState(null);
  const [imageMode, setImageMode] = useState('gradcam'); // 'gradcam' or 'vit'
  
  // AI Q&A State
  const [aiQuestion, setAiQuestion] = useState('');
  const [aiAnswer, setAiAnswer] = useState('');
  const [aiLoading, setAiLoading] = useState(false);

  useEffect(() => {
    if (activeTab !== 'image') {
      fetchExplanation(activeTab === 'summary' ? 'feature_importance' : activeTab);
    }
  }, [activeTab]);

  const fetchExplanation = async (type) => {
    try {
      setLoading(true);
      setError('');
      const res = await API.post('/model-explain', { type });
      if (res.data.error) {
        setError(res.data.error);
      } else {
        setData(res.data);
      }
    } catch (err) {
      setError('Failed to fetch explanation. Ensure a tabular model is trained.');
    } finally {
      setLoading(false);
    }
  };

  const handleAskAI = async () => {
    if (!aiQuestion.trim()) return;
    try {
      setAiLoading(true);
      const res = await API.post('/ask-model', { question: aiQuestion });
      setAiAnswer(res.data.answer);
    } catch (err) {
      setAiAnswer('Technical synchronization error. AI unavailable.');
    } finally {
      setAiLoading(false);
    }
  };

  const handleImageExplain = async () => {
    if (!selectedFile) return;
    try {
      setLoading(true);
      setError('');
      const formData = new FormData();
      formData.append('file', selectedFile);
      
      const endpoint = imageMode === 'gradcam' ? '/explain-image' : '/explain-vit';
      const res = await API.post(endpoint, formData);
      
      if (res.data.error) {
        setError(res.data.error);
      } else {
        setImageResult(res.data.gradcam || res.data.attention);
      }
    } catch (err) {
      setError('Image explanation failed. Ensure an image model is trained.');
    } finally {
      setLoading(false);
    }
  };

  const renderContent = () => {
    if (loading && !data && activeTab !== 'image') return <div className="flex items-center justify-center h-64 font-mono text-white/40">Loading Model Logic...</div>;
    
    switch (activeTab) {
      case 'summary':
        return (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <motion.div variants={fadeUp} className="border border-white/[.08] bg-[#111] p-6">
              <h3 className="font-display text-sm font-bold uppercase text-[#B7FF4A] mb-4 flex items-center gap-2">
                <Brain size={16} /> Model Identity
              </h3>
              <div className="space-y-4">
                <InfoItem label="Architecture" value={data?.summary?.model_name || 'Calculating...'} />
                <InfoItem label="Domain" value={data?.summary?.problem_type || 'General'} />
                <InfoItem label="Feature Count" value={data?.summary?.features_count || '0'} />
              </div>
            </motion.div>

            <motion.div variants={fadeUp} className="border border-white/[.08] bg-[#111] p-6">
              <h3 className="font-display text-sm font-bold uppercase text-[#6AA7FF] mb-4 flex items-center gap-2">
                <BarChart3 size={16} /> Global Feature Impact
              </h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={data?.feature_importance?.slice(0, 8)}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#222" vertical={false} />
                    <XAxis dataKey="feature" hide />
                    <YAxis stroke="#444" fontSize={10} fontFamily="monospace" />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#111', border: '1px solid rgba(255,255,255,0.1)', fontSize: '10px' }}
                      itemStyle={{ color: '#B7FF4A' }}
                    />
                    <Bar dataKey="importance" fill="#B7FF4A" radius={[2, 2, 0, 0]}>
                       {data?.feature_importance?.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={index % 2 === 0 ? '#B7FF4A' : '#6AA7FF'} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </motion.div>

            {/* AI Insight Card */}
            {data?.summary?.ai_summary && (
              <motion.div variants={fadeUp} className="md:col-span-2 border border-[#B7FF4A]/20 bg-[#B7FF4A]/5 p-6 rounded-xl">
                 <div className="flex items-center gap-2 mb-3">
                    <Brain size={18} className="text-[#B7FF4A]" />
                    <h3 className="font-display text-sm font-bold uppercase text-[#B7FF4A]">Neural Insight (Gemini AI)</h3>
                 </div>
                 <p className="font-mono text-[11px] text-white/70 leading-relaxed whitespace-pre-wrap italic">
                    {data.summary.ai_summary}
                 </p>
              </motion.div>
            )}

            {/* Ask AI Interface */}
            <motion.div variants={fadeUp} className="md:col-span-2 border border-white/[.08] bg-[#000] p-6 rounded-xl overflow-hidden relative">
               <div className="flex items-center justify-between mb-4">
                  <h3 className="font-display text-sm font-bold uppercase text-white flex items-center gap-2">
                    <Zap size={16} className="text-[#B7FF4A]" /> Deep Intelligence Query
                  </h3>
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-[#B7FF4A] animate-pulse" />
                    <span className="font-mono text-[8px] text-white/30 uppercase tracking-[0.2em]">Neural Link: Active</span>
                  </div>
               </div>
               
               <div className="flex flex-col md:flex-row gap-4">
                  <input 
                    type="text"
                    value={aiQuestion}
                    onChange={(e) => setAiQuestion(e.target.value)}
                    placeholder="Ask anything about how this model works..."
                    className="flex-1 bg-white/5 border border-white/10 px-4 py-3 font-mono text-[11px] text-white focus:outline-none focus:border-[#B7FF4A]/40 transition-all"
                  />
                  <button 
                    onClick={handleAskAI}
                    disabled={aiLoading}
                    className="px-6 py-3 bg-[#B7FF4A] text-black font-mono text-[10px] font-bold uppercase tracking-widest hover:brightness-110 disabled:opacity-50 transition-all shrink-0"
                  >
                    {aiLoading ? 'Interrogating Model...' : 'Query AI'}
                  </button>
               </div>

               {aiAnswer && (
                 <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} className="mt-6 pt-6 border-t border-white/5">
                    <div className="flex gap-4">
                       <div className="w-8 h-8 bg-[#B7FF4A]/10 border border-[#B7FF4A]/20 flex items-center justify-center shrink-0">
                          <Brain size={14} className="text-[#B7FF4A]" />
                       </div>
                       <p className="font-mono text-[11px] text-white/60 leading-relaxed">
                          {aiAnswer}
                       </p>
                    </div>
                 </motion.div>
               )}
            </motion.div>
          </div>
        );

      case 'shap':
        return (
          <motion.div variants={fadeUp} className="border border-white/[.08] bg-[#111] p-6">
             <div className="flex justify-between items-center mb-6">
              <h3 className="font-display text-sm font-bold uppercase text-white flex items-center gap-2">
                <Zap size={16} className="text-[#B7FF4A]" /> SHAP Strategic Values
              </h3>
              <div className="font-mono text-[9px] text-white/30 uppercase tracking-widest border border-white/10 px-2 py-1 rounded">
                Global Contribution
              </div>
            </div>
            <div className="h-[400px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={data?.shap} layout="vertical" margin={{ left: 40 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#222" horizontal={false} />
                  <XAxis type="number" stroke="#444" fontSize={10} />
                  <YAxis dataKey="feature" type="category" stroke="#666" fontSize={10} width={100} fontFamily="monospace" />
                  <Tooltip contentStyle={{ backgroundColor: '#000', border: '1px solid #333' }} />
                  <Bar dataKey="value" fill="#B7FF4A" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </motion.div>
        );

      case 'metrics':
        const isRegression = !!data?.metrics || !!data?.residuals;
        return (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
             <motion.div variants={fadeUp} className="border border-white/[.08] bg-[#111] p-6">
               <h3 className="font-display text-sm font-bold uppercase text-white mb-6">
                 {isRegression ? 'Residual Distribution' : 'ROC Curve (Convergence)'}
               </h3>
               <div className="h-64">
                 <ResponsiveContainer width="100%" height="100%">
                   {isRegression ? (
                     <AreaChart data={data?.residuals?.map((r, i) => ({ i, residual: r }))}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#222" />
                        <XAxis dataKey="i" hide />
                        <YAxis stroke="#444" fontSize={10} />
                        <Tooltip contentStyle={{ backgroundColor: '#000', border: '1px solid #333' }} />
                        <Area type="monotone" dataKey="residual" stroke="#6AA7FF" fill="#6AA7FF" fillOpacity={0.1} />
                     </AreaChart>
                   ) : (
                     <AreaChart data={data?.roc}>
                       <CartesianGrid strokeDasharray="3 3" stroke="#222" />
                       <XAxis dataKey="fpr" label={{ value: 'FPR', position: 'insideBottom', offset: -5, fontSize: 10, fill: '#444' }} stroke="#444" fontSize={10} />
                       <YAxis label={{ value: 'TPR', angle: -90, position: 'insideLeft', fontSize: 10, fill: '#444' }} stroke="#444" fontSize={10} />
                       <Tooltip />
                       <Area type="monotone" dataKey="tpr" stroke="#B7FF4A" fill="#B7FF4A" fillOpacity={0.1} />
                       <Line type="monotone" dataKey="fpr" stroke="#222" strokeDasharray="5 5" />
                     </AreaChart>
                   )}
                 </ResponsiveContainer>
               </div>
             </motion.div>

             <motion.div variants={fadeUp} className="border border-white/[.08] bg-[#111] p-6">
               <h3 className="font-display text-sm font-bold uppercase text-white mb-6">
                 {isRegression ? 'Regression Coefficients' : 'Confusion Density Map'}
               </h3>
               {isRegression ? (
                 <div className="space-y-4">
                   <div className="grid grid-cols-2 gap-4">
                      <div className="bg-white/5 p-4 border border-white/10 text-center">
                        <div className="font-mono text-[9px] text-white/30 uppercase mb-1">MSE</div>
                        <div className="font-mono text-xl text-[#FF6B6B] font-bold">{data?.metrics?.mse?.toFixed(4) || '0.0000'}</div>
                      </div>
                      <div className="bg-white/5 p-4 border border-white/10 text-center">
                        <div className="font-mono text-[9px] text-white/30 uppercase mb-1">R² Score</div>
                        <div className="font-mono text-xl text-[#B7FF4A] font-bold">{data?.metrics?.r2?.toFixed(4) || '0.0000'}</div>
                      </div>
                   </div>
                   <p className="font-mono text-[10px] text-white/40 leading-relaxed text-center py-4 italic">
                     "Residuals show the difference between actual and predicted values. Low variance around zero indicates high model stability."
                   </p>
                 </div>
               ) : data?.confusion_matrix ? (
                 <div className="grid grid-cols-2 gap-2 max-w-[300px] mx-auto">
                    {data.confusion_matrix.map((row, i) => 
                      row.map((val, j) => (
                        <div key={`${i}-${j}`} className={`aspect-square flex flex-col items-center justify-center p-4 border border-white/5 ${i === j ? 'bg-[#B7FF4A]/10' : 'bg-red-500/5'}`}>
                           <span className="font-mono text-2xl font-bold text-white">{val}</span>
                           <span className="font-mono text-[9px] text-white/30 uppercase mt-2">
                             {i===j?'Correct':'Miss'}
                           </span>
                        </div>
                      ))
                    )}
                 </div>
               ) : (
                <div className="text-center py-12 font-mono text-white/20 uppercase tracking-widest text-[10px]">Matrix data not available</div>
               )}
             </motion.div>
          </div>
        );

      case 'image':
        return (
          <div className="space-y-6">
            <div className="flex border-b border-white/10 mb-6">
              {['gradcam', 'vit'].map(m => (
                <button 
                  key={m} 
                  onClick={() => setImageMode(m)}
                  className={`px-6 py-3 font-mono text-[10px] uppercase tracking-widest transition-all ${imageMode === m ? 'text-[#B7FF4A] border-b-2 border-[#B7FF4A]' : 'text-white/40 hover:text-white/60'}`}
                >
                  {m === 'gradcam' ? 'Grad-CAM (CNN)' : 'Attention Map (ViT)'}
                </button>
              ))}
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
               <div className="border border-dashed border-white/10 p-8 flex flex-col items-center justify-center bg-[#111]">
                  <input type="file" id="img-upload" className="hidden" onChange={(e) => setSelectedFile(e.target.files[0])} />
                  <label htmlFor="img-upload" className="cursor-pointer group flex flex-col items-center">
                    <div className="w-16 h-16 rounded-full bg-white/5 flex items-center justify-center mb-4 group-hover:bg-[#B7FF4A]/20 transition-all">
                      <Upload className="text-white/40 group-hover:text-[#B7FF4A]" />
                    </div>
                    <span className="font-mono text-[11px] text-white/60 uppercase">
                      {selectedFile ? selectedFile.name : 'Target Image'}
                    </span>
                  </label>
                  
                  {selectedFile && (
                    <button 
                      onClick={handleImageExplain}
                      disabled={loading}
                      className="mt-6 px-10 py-3 bg-[#B7FF4A] text-black font-mono text-[11px] font-bold uppercase tracking-tighter hover:brightness-110 disabled:opacity-50"
                    >
                      {loading ? 'Processing Neural Paths...' : 'Generate XAI Overlay'}
                    </button>
                  )}
               </div>

               <div className="border border-white/[.08] bg-[#000] p-4 flex items-center justify-center min-h-[300px] relative overflow-hidden">
                  {imageResult ? (
                    <img src={`${API_BASE}${imageResult}`} alt="XAI Overlay" className="max-w-full h-auto rounded border border-white/10 shadow-2xl" />
                  ) : (
                    <div className="text-center space-y-4">
                      <ImageIcon className="mx-auto text-white/5" size={64} />
                      <p className="font-mono text-[10px] text-white/20 uppercase">Waiting for Neural Hook...</p>
                    </div>
                  )}
                  {loading && (
                    <div className="absolute inset-0 bg-black/40 backdrop-blur-sm flex items-center justify-center">
                       <div className="w-8 h-8 border-2 border-[#B7FF4A] border-t-transparent animate-spin rounded-full" />
                    </div>
                  )}
               </div>
            </div>
          </div>
        );

      case 'analytics':
        const logs = data?.training_logs || {};
        const activeModelName = data?.summary?.model_name || Object.keys(logs)[0];
        const logData = logs[activeModelName] || { loss: [], accuracy: [] };
        
        const chartData = logData.loss.map((l, i) => ({
          epoch: i + 1,
          loss: l,
          accuracy: logData.accuracy[i] || 0
        }));

        const handleDownload = async () => {
          try {
            const res = await API.get('/download-code');
            const blob = new Blob([res.data.code], { type: 'text/plain' });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = res.data.filename || 'train_local.py';
            a.click();
          } catch (err) {
            console.error('Download failed', err);
          }
        };

        return (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <motion.div variants={fadeUp} className="border border-white/[.08] bg-[#111] p-6">
                <h3 className="font-display text-sm font-bold uppercase text-[#FF6B6B] mb-6 flex items-center gap-2">
                  <Activity size={16} /> Loss Trajectory
                </h3>
                <div className="h-64">
                   <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#222" vertical={false} />
                        <XAxis dataKey="epoch" stroke="#444" fontSize={10} label={{ value: 'Epochs', position: 'insideBottom', offset: -5, fontSize: 8, fill: '#444' }} />
                        <YAxis stroke="#444" fontSize={10} />
                        <Tooltip contentStyle={{ backgroundColor: '#000', border: '1px solid #333', fontSize: '10px' }} />
                        <Line type="monotone" dataKey="loss" stroke="#FF6B6B" strokeWidth={2} dot={{ fill: '#FF6B6B' }} />
                      </LineChart>
                   </ResponsiveContainer>
                </div>
              </motion.div>

              <motion.div variants={fadeUp} className="border border-white/[.08] bg-[#111] p-6">
                <h3 className="font-display text-sm font-bold uppercase text-[#B7FF4A] mb-6 flex items-center gap-2">
                   <Target size={16} /> Accuracy Convergence
                </h3>
                <div className="h-64">
                   <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#222" vertical={false} />
                        <XAxis dataKey="epoch" stroke="#444" fontSize={10} label={{ value: 'Epochs', position: 'insideBottom', offset: -5, fontSize: 8, fill: '#444' }} />
                        <YAxis stroke="#444" fontSize={10} />
                        <Tooltip contentStyle={{ backgroundColor: '#000', border: '1px solid #333', fontSize: '10px' }} />
                        <Line type="monotone" dataKey="accuracy" stroke="#B7FF4A" strokeWidth={2} dot={{ fill: '#B7FF4A' }} />
                      </LineChart>
                   </ResponsiveContainer>
                </div>
              </motion.div>
            </div>

            {/* Neural Artifacts / Download Section */}
            <motion.div variants={fadeUp} className="border border-[#B7FF4A]/20 bg-[#B7FF4A]/5 p-8 flex flex-col md:flex-row items-center justify-between gap-6">
               <div className="flex gap-4">
                  <div className="w-12 h-12 bg-[#B7FF4A]/20 flex items-center justify-center rounded-lg">
                    <Zap className="text-[#B7FF4A]" size={24} />
                  </div>
                  <div>
                    <h4 className="font-display text-lg font-bold text-white uppercase tracking-tight">Neural Export (Local SDK)</h4>
                    <p className="font-mono text-[10px] text-white/40 max-w-md">
                      Generate a standalone Python environment with the exact architecture and parameters discovered during this AutoML cycle.
                    </p>
                  </div>
               </div>
               <button 
                onClick={handleDownload}
                className="px-8 py-4 bg-[#B7FF4A] text-black font-mono text-[11px] font-bold uppercase tracking-widest hover:scale-105 transition-all flex items-center gap-2 shadow-[0_0_20px_rgba(183,255,74,0.3)]"
               >
                 <Upload size={14} className="rotate-180" /> Download Neural Script
               </button>
            </motion.div>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <DashboardLayout>
      <div className="max-w-7xl mx-auto space-y-8">
        <header className="flex flex-col md:flex-row md:items-end justify-between gap-6 border-b border-white/[.08] pb-8">
          <div>
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 bg-[#B7FF4A] flex items-center justify-center">
                <Search size={20} className="text-black" />
              </div>
              <h1 className="font-display text-4xl font-bold uppercase tracking-tight text-white">XAI Module</h1>
            </div>
            <p className="font-mono text-[11px] text-white/40 tracking-wider uppercase ml-1">
              Neural Network Interpretation & Feature Attribution
            </p>
          </div>
          
          <div className="flex gap-1 bg-[#111] p-1 border border-white/5">
            {['summary', 'shap', 'metrics', 'analytics', 'image'].map(tab => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`px-4 py-2 font-mono text-[10px] uppercase tracking-wider transition-all ${
                  activeTab === tab ? 'bg-[#B7FF4A] text-black font-bold' : 'text-white/40 hover:bg-white/5'
                }`}
              >
                {tab}
              </button>
            ))}
          </div>
        </header>

        {error && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex items-center gap-3 p-4 bg-red-500/10 border border-red-500/20 text-red-500 font-mono text-[11px] uppercase">
            <AlertCircle size={14} /> {error}
          </motion.div>
        )}

        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            transition={{ duration: 0.3 }}
          >
            {renderContent()}
          </motion.div>
        </AnimatePresence>
      </div>
    </DashboardLayout>
  );
}

function InfoItem({ label, value }) {
  return (
    <div>
      <p className="font-mono text-[9px] text-white/30 uppercase tracking-[0.2em] mb-1">{label}</p>
      <p className="font-mono text-[13px] text-white/90 font-bold">{value}</p>
    </div>
  );
}
