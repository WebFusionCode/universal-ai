import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import DashboardLayout from '../components/DashboardLayout';
import API from '../lib/api';

const fadeUp = { hidden: { opacity: 0, y: 20 }, visible: { opacity: 1, y: 0 } };

export default function ModelExplain() {
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadModels();
  }, []);

  const loadModels = async () => {
    try {
      setLoading(true);
      const res = await API.get('/models').catch(() => ({ data: { models: [] } }));
      setModels(res.data.models || []);
    } catch (error) {
      console.error('Error loading models:', error);
    } finally {
      setLoading(false);
    }
  };

  const explainModel = async (modelId) => {
    try {
      setLoading(true);
      const res = await API.get(`/models/${modelId}/explain`).catch(() => ({ 
        data: { explanation: 'Model explanation not available' } 
      }));
      setExplanation(res.data.explanation || 'No explanation available');
    } catch (error) {
      console.error('Error explaining model:', error);
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
            Model Explanations
          </h1>
          <p className="font-mono text-[11px] text-white/40 tracking-wider uppercase">
            Understand & Interpret Your AI Models
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Models List */}
          <motion.div
            variants={fadeUp}
            className="border border-white/[.06] p-6"
          >
            <h2 className="font-display text-lg font-bold text-white mb-4 uppercase">Models</h2>
            <div className="space-y-2">
              {models.length === 0 ? (
                <p className="font-mono text-[11px] text-white/30">No models available</p>
              ) : (
                models.map((model, idx) => (
                  <motion.button
                    key={idx}
                    whileHover={{ paddingLeft: '1rem' }}
                    onClick={() => {
                      setSelectedModel(model);
                      explainModel(model.id);
                    }}
                    className={`w-full text-left px-3 py-2 border border-transparent transition-all duration-300 ${
                      selectedModel?.id === model.id
                        ? 'border-[#B7FF4A] bg-[#B7FF4A]/5 text-[#B7FF4A]'
                        : 'border-white/[.06] text-white/60 hover:border-white/[.12]'
                    }`}
                  >
                    <p className="font-mono text-[11px] font-bold uppercase truncate">
                      {model.name || 'Untitled Model'}
                    </p>
                  </motion.button>
                ))
              )}
            </div>
          </motion.div>

          {/* Explanation Panel */}
          <motion.div
            variants={fadeUp}
            className="lg:col-span-2 border border-white/[.06] p-6"
          >
            <h2 className="font-display text-lg font-bold text-white mb-4 uppercase">
              {selectedModel ? 'Model Details' : 'Select a Model'}
            </h2>

            {selectedModel && explanation && (
              <div className="space-y-4">
                <div>
                  <p className="font-mono text-[10px] text-white/30 mb-2 uppercase tracking-[0.1em]">
                    Name
                  </p>
                  <p className="font-display text-lg text-white">{selectedModel.name}</p>
                </div>
                <div>
                  <p className="font-mono text-[10px] text-white/30 mb-2 uppercase tracking-[0.1em]">
                    Type
                  </p>
                  <p className="font-display text-lg text-white">{selectedModel.type || 'Neural Network'}</p>
                </div>
                <div>
                  <p className="font-mono text-[10px] text-white/30 mb-2 uppercase tracking-[0.1em]">
                    Explanation
                  </p>
                  <p className="font-mono text-[11px] text-white/60 leading-relaxed">{explanation}</p>
                </div>
              </div>
            )}

            {!selectedModel && (
              <p className="font-mono text-[11px] text-white/30 text-center py-8">
                Select a model to view its explanation
              </p>
            )}
          </motion.div>
        </div>
      </motion.div>
    </DashboardLayout>
  );
}
