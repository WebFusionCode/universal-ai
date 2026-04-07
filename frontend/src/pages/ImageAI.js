import React, { useState } from 'react';
import { motion } from 'framer-motion';
import DashboardLayout from '../components/DashboardLayout';
import API from '../lib/api';

const fadeUp = { hidden: { opacity: 0, y: 20 }, visible: { opacity: 1, y: 0 } };

export default function ImageAI() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [preview, setPreview] = useState(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      const reader = new FileReader();
      reader.onloadend = () => setPreview(reader.result);
      reader.readAsDataURL(selectedFile);
    }
  };

  const handleGenerateImage = async () => {
    if (!file) {
      alert('Please select an image');
      return;
    }

    try {
      setLoading(true);
      const formData = new FormData();
      formData.append('file', file);

      const res = await API.post('/image-ai/process', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      }).catch(() => ({
        data: { result: 'Image processing not available', prediction: 'Processing unavailable' }
      }));

      setResult(res.data);
    } catch (error) {
      console.error('Error processing image:', error);
      alert('Error processing image');
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
            Image AI
          </h1>
          <p className="font-mono text-[11px] text-white/40 tracking-wider uppercase">
            Advanced Image Analysis & Generation
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Upload Section */}
          <motion.div
            variants={fadeUp}
            className="border border-white/[.06] p-8"
          >
            <h2 className="font-display text-lg font-bold text-white mb-6 uppercase">Upload Image</h2>

            <div
              onClick={() => document.getElementById('image-input').click()}
              className="border-2 border-dashed border-white/[.06] rounded hover:border-white/[.12] transition-all duration-300 p-8 text-center cursor-pointer group"
            >
              {preview ? (
                <img
                  src={preview}
                  alt="Preview"
                  className="max-h-64 mx-auto rounded"
                />
              ) : (
                <div className="py-8">
                  <p className="font-display text-lg text-white mb-2">Drop image here</p>
                  <p className="font-mono text-[11px] text-white/30">or click to browse</p>
                </div>
              )}
            </div>

            <input
              id="image-input"
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              className="hidden"
            />

            <button
              onClick={handleGenerateImage}
              disabled={loading || !file}
              className="w-full mt-6 px-6 py-3 bg-[#B7FF4A] text-black font-bold uppercase tracking-wider disabled:opacity-50 hover:bg-[#B7FF4A]/90 transition-all duration-300"
            >
              {loading ? 'Processing...' : 'Analyze Image'}
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
                <div>
                  <p className="font-mono text-[10px] text-white/30 mb-2 uppercase tracking-[0.1em]">
                    Analysis
                  </p>
                  <p className="font-mono text-[11px] text-white/60">{result.result || 'Processing complete'}</p>
                </div>
                {result.prediction && (
                  <div>
                    <p className="font-mono text-[10px] text-white/30 mb-2 uppercase tracking-[0.1em]">
                      Prediction
                    </p>
                    <p className="font-mono text-[11px] text-white/60">{result.prediction}</p>
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
