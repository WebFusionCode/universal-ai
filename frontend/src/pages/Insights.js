import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import DashboardLayout from '../components/DashboardLayout';
import API from '../lib/api';

const fadeUp = { hidden: { opacity: 0, y: 20 }, visible: { opacity: 1, y: 0 } };

export default function Insights() {
  const [insights, setInsights] = useState([]);
  const [loading, setLoading] = useState(true);
  const [metrics, setMetrics] = useState({
    avgAccuracy: 0,
    totalExperiments: 0,
    bestModel: '-',
    improvementRate: 0
  });

  useEffect(() => {
    loadInsights();
  }, []);

  const loadInsights = async () => {
    try {
      setLoading(true);
      const [insRes, metricsRes] = await Promise.all([
        API.get('/insights').catch(() => ({ data: { insights: [] } })),
        API.get('/metrics').catch(() => ({ data: metrics }))
      ]);

      setInsights(insRes.data.insights || []);
      setMetrics(metricsRes.data || metrics);
    } catch (error) {
      console.error('Error loading insights:', error);
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
            Insights
          </h1>
          <p className="font-mono text-[11px] text-white/40 tracking-wider uppercase">
            AI-Driven Analytics & Recommendations
          </p>
        </div>

        {/* Metrics Grid */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {[
            { label: 'Avg Accuracy', value: `${metrics.avgAccuracy}%`, color: '#B7FF4A' },
            { label: 'Total Experiments', value: metrics.totalExperiments, color: '#6AA7FF' },
            { label: 'Best Model', value: metrics.bestModel, color: '#FF6B9D' },
            { label: 'Improvement', value: `${metrics.improvementRate}%`, color: '#4FD1C5' }
          ].map((metric, idx) => (
            <motion.div
              key={idx}
              variants={fadeUp}
              className="border border-white/[.06] p-6 hover:border-white/[.12] transition-all duration-300"
            >
              <p className="font-mono text-[10px] tracking-[0.15em] uppercase text-white/30 mb-2">
                {metric.label}
              </p>
              <p className="font-display text-2xl font-bold" style={{ color: metric.color }}>
                {metric.value}
              </p>
            </motion.div>
          ))}
        </div>

        {/* Insights List */}
        <motion.div
          variants={fadeUp}
          className="border border-white/[.06] p-6"
        >
          <h2 className="font-display text-lg font-bold text-white mb-6 uppercase">Key Insights</h2>

          {insights.length === 0 ? (
            <p className="font-mono text-[11px] text-white/30 text-center py-8">
              Run some experiments to generate insights
            </p>
          ) : (
            <div className="space-y-4">
              {insights.map((insight, idx) => (
                <motion.div
                  key={idx}
                  variants={fadeUp}
                  className="border-l-2 border-[#B7FF4A] pl-4 py-2"
                >
                  <p className="font-display text-white mb-1">{insight.title}</p>
                  <p className="font-mono text-[11px] text-white/60">{insight.description}</p>
                </motion.div>
              ))}
            </div>
          )}
        </motion.div>
      </motion.div>
    </DashboardLayout>
  );
}
