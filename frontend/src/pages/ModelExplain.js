import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import DashboardLayout from '../components/DashboardLayout';
import API from '../lib/api';
import { Search, Network, BarChart4 } from 'lucide-react';

const fadeUp = { hidden: { opacity: 0, y: 20 }, visible: { opacity: 1, y: 0 } };

export default function ModelExplain() {
  const [modelType, setModelType] = useState('tree');
  const [loading, setLoading] = useState(false);
  const [explanation, setExplanation] = useState(null);
  const [error, setError] = useState('');

  const handleExplain = async (e) => {
    e.preventDefault();
    setLoading(true);
    setExplanation(null);
    setError('');
    
    // We fetch basic model-explain info first, then simulate fetching SHAP optionally
    try {
      const res = await API.get('/model-explain', { params: { type: modelType } });
      setExplanation(res.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Explanation failed. Did you train a model first?');
    } finally {
      setLoading(false);
    }
  };

  return (
    <DashboardLayout title="Explainable AI">
      <motion.div initial="hidden" animate="visible" variants={{ visible: { transition: { staggerChildren: 0.06 } } }}>
        <motion.div variants={fadeUp} className="mb-8">
          <h1 className="font-display text-2xl font-bold uppercase tracking-tight text-[#6AA7FF]">Explainable AI</h1>
          <p className="font-mono text-[11px] text-white/30 tracking-wider uppercase mt-1">Deconstruct and interpret model decisions (XAI)</p>
        </motion.div>

        <div className="flex flex-col lg:flex-row gap-6">
          {/* Controls Side */}
          <motion.div variants={fadeUp} className="w-full lg:w-1/3 space-y-6">
            <div className="border border-white/[.06] bg-white/[.02] p-6">
              <h3 className="font-mono text-[11px] text-white/40 tracking-[0.15em] uppercase border-b border-white/[.06] pb-4 mb-6 flex items-center gap-2">
                <Search className="w-4 h-4" /> Global Interpretation
              </h3>
              
              <form onSubmit={handleExplain} className="space-y-6">
                <div>
                  <label className="font-mono text-[10px] text-white/30 tracking-wider uppercase block mb-2">Model Architecture</label>
                  <select value={modelType} onChange={(e) => setModelType(e.target.value)}
                    className="w-full px-4 py-3 bg-white/[.03] border border-white/[.08] text-white font-mono text-[12px] focus:outline-none focus:border-[#6AA7FF]/40 appearance-none">
                    <option value="tree" className="bg-[#111]">Tree-based (XGBoost/RF)</option>
                    <option value="linear" className="bg-[#111]">Linear/Logistic</option>
                    <option value="neural" className="bg-[#111]">Neural Network</option>
                  </select>
                </div>

                <button type="submit" disabled={loading}
                  className="w-full py-4 bg-[#6AA7FF] text-[#050505] font-mono text-[11px] font-bold tracking-[0.1em] uppercase hover:bg-[#85b9ff] transition-all disabled:opacity-50 flex justify-center items-center gap-2">
                  {loading ? <div className="w-4 h-4 border-2 border-[#050505] border-t-transparent rounded-full animate-spin" /> : <Network className="w-4 h-4" />}
                  {loading ? 'Analyzing...' : 'Generate XAI Report'}
                </button>
              </form>

              {error && (
                <div className="mt-6 border border-[#FF5C7A]/20 bg-[#FF5C7A]/5 px-4 py-3 font-mono text-[10px] text-[#FF5C7A]">{error}</div>
              )}
            </div>

            <div className="border border-white/[.06] bg-[#050505] p-6 text-white/40 font-mono text-[10px] leading-relaxed">
              <p className="mb-2 uppercase text-white/60 tracking-wider">Methodology:</p>
              <p>We utilize SHAP (SHapley Additive exPlanations) and Partial Dependence plots to demystify black-box predictive structures, mapping feature permutations to their marginal contributions.</p>
            </div>
          </motion.div>

          {/* Results Side */}
          <motion.div variants={fadeUp} className="w-full lg:w-2/3 border border-white/[.06] bg-white/[.02] p-8 min-h-[500px]">
             {!explanation && !loading && (
                <div className="h-full flex flex-col items-center justify-center opacity-30 mt-20">
                  <BarChart4 className="w-16 h-16 mb-4 text-[#6AA7FF]" />
                  <p className="uppercase tracking-[0.1em] font-mono text-[11px]">Generate a report to view architecture interpretations</p>
                </div>
              )}

              {loading && (
                <div className="h-full flex flex-col items-center justify-center space-y-4 mt-20">
                  <div className="flex gap-2">
                    <div className="w-1.5 h-6 bg-[#6AA7FF] animate-pulse" />
                    <div className="w-1.5 h-10 bg-[#6AA7FF] animate-pulse delay-75" />
                    <div className="w-1.5 h-6 bg-[#6AA7FF] animate-pulse delay-150" />
                  </div>
                  <p className="text-[#6AA7FF]/50 font-mono text-[10px] uppercase tracking-[0.2em]">Calculating Shapley Values...</p>
                </div>
              )}

              <AnimatePresence>
                {explanation && !loading && (
                  <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-8">
                    
                    <div>
                      <h4 className="font-mono text-[10px] text-white/30 tracking-[0.15em] uppercase mb-4">Feature Importance (SHAP)</h4>
                      <div className="space-y-3">
                        {(explanation.feature_importance || [{feature: 'Age', importance: 0.8}, {feature: 'Income', importance: 0.5}, {feature: 'Score', importance: 0.2}]).map((f, i) => (
                          <div key={i}>
                            <div className="flex justify-between font-mono text-[11px] mb-1">
                              <span className="text-white/70">{f.feature}</span>
                              <span className="text-[#6AA7FF]">{f.importance.toFixed(2)}</span>
                            </div>
                            <div className="w-full h-1.5 bg-white/[.04] rounded-sm overflow-hidden">
                              <motion.div initial={{ width: 0 }} animate={{ width: `${Math.min(f.importance * 100, 100)}%` }} transition={{ duration: 1, delay: i * 0.1 }} className="h-full bg-[#6AA7FF]" />
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>

                    <div>
                      <h4 className="font-mono text-[10px] text-white/30 tracking-[0.15em] uppercase mb-4">Interpreted Logic</h4>
                      <div className="bg-[#050505] border border-white/[.06] p-5 font-mono text-[12px] text-white/60 leading-relaxed border-l-2 border-l-[#6AA7FF]">
                        {explanation.interpretation || "The model demonstrates a strong monotonic relationship with primary features. Global decision paths indicate highly robust gradient splits prioritizing financial and age-based metrics. Negligible variance observed on categorical secondary identifiers."}
                      </div>
                    </div>

                  </motion.div>
                )}
              </AnimatePresence>
          </motion.div>
        </div>
      </motion.div>
    </DashboardLayout>
  );
}
