import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import DashboardLayout from '../components/DashboardLayout';
import API from '../lib/api';
import { AudioLines, Image as ImageIcon, Video, Wand2 } from 'lucide-react';

const fadeUp = { hidden: { opacity: 0, y: 20 }, visible: { opacity: 1, y: 0 } };

export default function GenerativeAI() {
  const [activeTab, setActiveTab] = useState('image');
  const [prompt, setPrompt] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const handleGenerate = async (e) => {
    e.preventDefault();
    if (!prompt.trim()) return;
    setLoading(true);
    setResult(null);
    setError('');
    try {
      if (activeTab === 'audio' || activeTab === 'video') {
        setResult({ message: "Coming soon" });
        setLoading(false);
        return;
      }

      const endpoint = '/generate-image';
      const res = await API.post(endpoint, { prompt });
      setResult(res.data);
    } catch (err) {
      setError(err.response?.data?.detail || err.response?.data?.error || 'Generation failed.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <DashboardLayout title="Generative Studio">
      <motion.div initial="hidden" animate="visible" variants={{ visible: { transition: { staggerChildren: 0.06 } } }}>
        <motion.div variants={fadeUp} className="mb-8">
          <h1 className="font-display text-2xl font-bold uppercase tracking-tight">Generative Studio</h1>
          <p className="font-mono text-[11px] text-white/30 tracking-wider uppercase mt-1">Multi-Modal AI Engine</p>
        </motion.div>

        {/* Tab Selection */}
        <motion.div variants={fadeUp} className="flex gap-4 mb-8">
          {[
            { id: 'image', label: 'Image Gen', icon: ImageIcon },
            { id: 'audio', label: 'Audio Synthesis', icon: AudioLines },
            { id: 'video', label: 'Video AI', icon: Video },
          ].map(tab => (
            <button key={tab.id} onClick={() => { setActiveTab(tab.id); setResult(null); setError(''); }}
              className={`flex flex-col items-center gap-2 p-5 border flex-1 transition-all ${activeTab === tab.id ? 'border-[#B7FF4A]/50 bg-[#B7FF4A]/5 shadow-[0_0_20px_rgba(183,255,74,0.1)]' : 'border-white/[.06] bg-[#111] hover:border-white/[.15] opacity-50 hover:opacity-100'}`}>
              <tab.icon className={`w-5 h-5 ${activeTab === tab.id ? 'text-[#B7FF4A]' : 'text-white/40'}`} />
              <span className={`font-mono text-[10px] tracking-[0.1em] uppercase ${activeTab === tab.id ? 'text-white' : 'text-white/40'}`}>{tab.label}</span>
            </button>
          ))}
        </motion.div>

        {/* Studio Canvas */}
        <motion.div variants={fadeUp} className="border border-white/[.06] bg-white/[.02] p-8">
          <form onSubmit={handleGenerate} className="flex gap-4 items-start">
            <div className="flex-1">
              <label className="font-mono text-[10px] text-white/30 tracking-[0.15em] uppercase mb-3 block">Synthesis Prompt</label>
              <textarea value={prompt} onChange={(e) => setPrompt(e.target.value)}
                className="w-full bg-[#050505] border border-white/[.08] p-4 font-mono text-[13px] text-white/80 placeholder-white/20 focus:outline-none focus:border-[#B7FF4A]/50 transition-all resize-none h-24"
                placeholder={`Describe the ${activeTab} you want to generate...`} />
            </div>
            <button type="submit" disabled={loading || !prompt.trim()}
              className="mt-7 flex flex-col items-center justify-center p-4 border border-[#B7FF4A]/30 bg-[#B7FF4A]/10 text-[#B7FF4A] h-24 min-w-[120px] hover:bg-[#B7FF4A]/20 transition-all disabled:opacity-30 disabled:cursor-not-allowed">
              {loading ? (
                <div className="w-5 h-5 border-2 border-[#B7FF4A] border-t-transparent rounded-full animate-spin" />
              ) : (
                <>
                  <Wand2 className="w-5 h-5 mb-2" />
                  <span className="font-mono text-[10px] font-bold tracking-[0.1em] uppercase">Generate</span>
                </>
              )}
            </button>
          </form>

          {/* Results Output */}
          <AnimatePresence>
            {error && (
              <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} className="mt-6 border border-[#FF5C7A]/20 bg-[#FF5C7A]/5 px-4 py-3 font-mono text-[11px] text-[#FF5C7A]">
                [SYS_ERR]: {error}
              </motion.div>
            )}

            {result && (
              <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="mt-8 border-t border-white/[.06] pt-8">
                <h3 className="font-mono text-[10px] text-[#B7FF4A] tracking-[0.15em] uppercase mb-4">Output Artifact</h3>
                <div className="bg-[#050505] border border-white/[.08] p-4 flex items-center justify-center min-h-[300px]">
                  
                  {activeTab === 'image' && result.image_url && (
                    <img src={result.image_url} alt="Generated UI" className="max-w-full max-h-[500px] border border-white/10" />
                  )}
                  {activeTab === 'image' && result.base64_data && (
                    <img src={`data:image/png;base64,${result.base64_data}`} alt="Generated UI" className="max-w-full max-h-[500px] border border-white/10" />
                  )}

                  {activeTab === 'audio' && (result.audio_url || result.base64_data) && (
                    <audio controls className="w-full max-w-md invert opacity-80" src={result.audio_url || `data:audio/mp3;base64,${result.base64_data}`} />
                  )}

                  {activeTab === 'video' && result.video_url && (
                    <video controls src={result.video_url} className="max-w-full max-h-[500px] border border-white/10" />
                  )}
                  
                  {activeTab !== 'image' && result.message === "Coming soon" && (
                    <div className="flex flex-col items-center justify-center opacity-50">
                      <Wand2 className="w-12 h-12 mb-4 text-white/50" />
                      <p className="uppercase tracking-[0.2em] font-mono text-[11px] text-white">Feature Coming Soon</p>
                    </div>
                  )}
                  {(!result.image_url && !result.base64_data && !result.audio_url && !result.video_url && result.message && result.message !== "Coming soon") && (
                    <p className="font-mono text-[12px] text-white/50">{result.message}</p>
                  )}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      </motion.div>
    </DashboardLayout>
  );
}
