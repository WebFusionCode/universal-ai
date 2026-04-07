import React, { useState } from 'react';
import { motion } from 'framer-motion';
import DashboardLayout from '../components/DashboardLayout';
import API from '../lib/api';

const fadeUp = { hidden: { opacity: 0, y: 20 }, visible: { opacity: 1, y: 0 } };

export default function AudioAI() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [fileName, setFileName] = useState('');

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setFileName(selectedFile.name);
    }
  };

  const handleProcessAudio = async () => {
    if (!file) {
      alert('Please select an audio file');
      return;
    }

    try {
      setLoading(true);
      const formData = new FormData();
      formData.append('file', file);

      const res = await API.post('/audio-ai/process', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      }).catch(() => ({
        data: { transcription: 'Audio processing not available', analysis: 'Processing unavailable' }
      }));

      setResult(res.data);
    } catch (error) {
      console.error('Error processing audio:', error);
      alert('Error processing audio');
    } finally {
      setLoading(false);
    }
  };

  return (
    <DashboardLayout>
      <motion.div
        variants={fadeUp}
        initial="hidden"
        animate="visible"
        className="space-y-8"
      >
        <div>
          <h1 className="font-display text-4xl font-bold uppercase tracking-tight text-white mb-2">
            Audio AI
          </h1>
          <p className="font-mono text-[11px] text-white/40 tracking-wider uppercase">
            Advanced Audio Analysis & Transcription
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Upload Section */}
          <motion.div
            variants={fadeUp}
            className="border border-white/[.06] p-8"
          >
            <h2 className="font-display text-lg font-bold text-white mb-6 uppercase">Upload Audio</h2>

            <div
              onClick={() => document.getElementById('audio-input').click()}
              className="border-2 border-dashed border-white/[.06] rounded hover:border-white/[.12] transition-all duration-300 p-8 text-center cursor-pointer group"
            >
              {fileName ? (
                <div className="py-4">
                  <p className="font-display text-lg text-white mb-2">📁 {fileName}</p>
                  <p className="font-mono text-[11px] text-white/30">Click to change</p>
                </div>
              ) : (
                <div className="py-8">
                  <p className="font-display text-lg text-white mb-2">Drop audio file here</p>
                  <p className="font-mono text-[11px] text-white/30">or click to browse</p>
                </div>
              )}
            </div>

            <input
              id="audio-input"
              type="file"
              accept="audio/*"
              onChange={handleFileChange}
              className="hidden"
            />

            <button
              onClick={handleProcessAudio}
              disabled={loading || !file}
              className="w-full mt-6 px-6 py-3 bg-[#6AA7FF] text-white font-bold uppercase tracking-wider disabled:opacity-50 hover:bg-[#6AA7FF]/90 transition-all duration-300"
            >
              {loading ? 'Processing...' : 'Analyze Audio'}
            </button>
          </motion.div>

          {/* Results Section */}
          {result && (
            <motion.div
              variants={fadeUp}
              className="border border-white/[.06] p-6"
            >
              <h2 className="font-display text-lg font-bold text-white mb-4 uppercase">Results</h2>
              <div className="space-y-4">
                {result.transcription && (
                  <div>
                    <p className="font-mono text-[10px] text-white/30 mb-2 uppercase tracking-[0.1em]">
                      Transcription
                    </p>
                    <p className="font-mono text-[11px] text-white/60">{result.transcription}</p>
                  </div>
                )}
                {result.analysis && (
                  <div>
                    <p className="font-mono text-[10px] text-white/30 mb-2 uppercase tracking-[0.1em]">
                      Analysis
                    </p>
                    <p className="font-mono text-[11px] text-white/60">{result.analysis}</p>
                  </div>
                )}
              </div>
            </motion.div>
          )}
        </div>
      </motion.div>
    </DashboardLayout>
  );
}
