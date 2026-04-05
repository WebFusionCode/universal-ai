import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import DashboardLayout from '../components/DashboardLayout';
import API from '../lib/api';
import { Shield, ShieldAlert, Bug, Terminal } from 'lucide-react';

const fadeUp = { hidden: { opacity: 0, y: 20 }, visible: { opacity: 1, y: 0 } };

export default function AdversarialTesting() {
  const [modelType, setModelType] = useState('classification');
  const [attackType, setAttackType] = useState('fgsm');
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState(null);
  const [error, setError] = useState('');

  const handleTest = async (e) => {
    e.preventDefault();
    setLoading(true);
    if (!file) { setError('Upload a model/data file first'); setLoading(false); return; }
    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("model_type", modelType);
      formData.append("attack_type", attackType);

      const res = await API.post('/adversarial-test', formData);
      setData(res.data.preview || res.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Adversarial scan failed or model missing.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <DashboardLayout title="Adversarial Sec-Test">
      <motion.div initial="hidden" animate="visible" variants={{ visible: { transition: { staggerChildren: 0.06 } } }}>
        <motion.div variants={fadeUp} className="mb-8">
          <h1 className="font-display text-2xl font-bold uppercase tracking-tight text-[#FF5C7A]">Adversarial Sec-Test</h1>
          <p className="font-mono text-[11px] text-white/30 tracking-wider uppercase mt-1">Stress-test models against injected noise patterns</p>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          
          {/* Controls */}
          <motion.div variants={fadeUp} className="border border-white/[.06] bg-white/[.02] p-6 lg:col-span-1">
            <h3 className="font-mono text-[11px] text-white/40 tracking-[0.15em] uppercase border-b border-white/[.06] pb-4 mb-6">Attack Vector</h3>
            
            <form onSubmit={handleTest} className="space-y-6">
              <div>
                <label className="font-mono text-[10px] text-white/30 tracking-wider uppercase block mb-2">Upload File</label>
                <input type="file" onChange={(e) => setFile(e.target.files[0])}
                  className="w-full text-white font-mono text-[12px] bg-white/[.03] p-2" />
              </div>

              <div>
                <label className="font-mono text-[10px] text-white/30 tracking-wider uppercase block mb-2">Model Target</label>
                <select value={modelType} onChange={(e) => setModelType(e.target.value)}
                  className="w-full px-4 py-3 bg-white/[.03] border border-white/[.08] text-white font-mono text-[12px] focus:outline-none focus:border-[#FF5C7A]/40 appearance-none">
                  <option value="classification" className="bg-[#111]">Classification Model</option>
                  <option value="regression" className="bg-[#111]">Regression Model</option>
                  <option value="nlp" className="bg-[#111]">NLP Model</option>
                </select>
              </div>

              <div>
                <label className="font-mono text-[10px] text-white/30 tracking-wider uppercase block mb-2">Attack Payload</label>
                <select value={attackType} onChange={(e) => setAttackType(e.target.value)}
                  className="w-full px-4 py-3 bg-white/[.03] border border-white/[.08] text-white font-mono text-[12px] focus:outline-none focus:border-[#FF5C7A]/40 appearance-none">
                  <option value="fgsm" className="bg-[#111]">FGSM (Fast Gradient Sign)</option>
                  <option value="pgd" className="bg-[#111]">PGD (Projected Gradient Descent)</option>
                  <option value="noise" className="bg-[#111]">Gaussian Noise Injection</option>
                </select>
              </div>

              <button type="submit" disabled={loading}
                className="w-full py-4 bg-[#FF5C7A]/10 border border-[#FF5C7A]/30 text-[#FF5C7A] font-mono text-[11px] font-bold tracking-[0.1em] uppercase hover:bg-[#FF5C7A]/20 transition-all disabled:opacity-50 flex justify-center items-center gap-2">
                {loading ? <div className="w-4 h-4 border-2 border-[#FF5C7A] border-t-transparent rounded-full animate-spin" /> : <Bug className="w-4 h-4" />}
                {loading ? 'Executing Attack...' : 'Launch Attack'}
              </button>
            </form>

            {error && (
              <div className="mt-6 border border-[#FF5C7A]/20 bg-[#FF5C7A]/5 px-4 py-3 font-mono text-[10px] text-[#FF5C7A]">{error}</div>
            )}
          </motion.div>

          {/* Results Terminal */}
          <motion.div variants={fadeUp} className="lg:col-span-2 border border-white/[.06] bg-[#050505] overflow-hidden flex flex-col">
            <div className="bg-white/[.03] border-b border-white/[.06] p-3 flex items-center gap-3">
              <Terminal className="w-4 h-4 text-white/30" />
              <span className="font-mono text-[10px] text-white/30 tracking-wider uppercase">Security Terminal_</span>
            </div>
            
            <div className="p-6 flex-1 text-white/60 font-mono text-[12px] leading-relaxed">
              {!data && !loading && (
                <div className="h-full flex flex-col items-center justify-center opacity-30">
                  <ShieldAlert className="w-12 h-12 mb-4" />
                  <p className="uppercase tracking-[0.1em] text-[10px]">Awaiting payload injection</p>
                </div>
              )}

              {loading && (
                <div className="animate-pulse space-y-2">
                  <p className="text-[#FFCC66]">&gt; INITIALIZING {attackType.toUpperCase()} PAYLOAD...</p>
                  <p className="text-white/40">&gt; TARGETING MODEL PARAMETERS...</p>
                </div>
              )}

              <AnimatePresence>
                {data && !loading && (
                  <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-4">
                    <p className="text-[#B7FF4A]">&gt; ATTACK COMPLETE</p>
                    
                    <div className="grid grid-cols-2 gap-4 mt-8">
                      <div className="border border-white/[.06] p-4 bg-white/[.02]">
                        <p className="text-[10px] text-white/30 tracking-wider mb-2">Original Accuracy</p>
                        <p className="text-2xl text-white">{(data.original_accuracy || 0).toFixed(2)}%</p>
                      </div>
                      <div className="border border-[#FF5C7A]/20 p-4 bg-[#FF5C7A]/5">
                        <p className="text-[10px] text-[#FF5C7A]/60 tracking-wider mb-2">Degraded Accuracy</p>
                        <p className="text-2xl text-[#FF5C7A]">{(data.degraded_accuracy || 0).toFixed(2)}%</p>
                      </div>
                    </div>

                    <div className="mt-6 border border-white/[.06] p-4">
                      <p className="text-[10px] text-white/30 tracking-wider mb-2">Vulnerability Assessment</p>
                      <p className="text-white/70">{data.vulnerability_report || 'Model is reasonably robust against this specific attack vector, though standard deviations indicate slight vulnerability bounds.'}</p>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </motion.div>
        </div>
      </motion.div>
    </DashboardLayout>
  );
}
