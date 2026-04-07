import React, { useState } from 'react';
import { motion } from 'framer-motion';
import DashboardLayout from '../components/DashboardLayout';
import API from '../lib/api';

const fadeUp = { hidden: { opacity: 0, y: 20 }, visible: { opacity: 1, y: 0 } };

export default function Compiler() {
  const [code, setCode] = useState('');
  const [language, setLanguage] = useState('python');
  const [output, setOutput] = useState('');
  const [loading, setLoading] = useState(false);

  const handleCompile = async () => {
    if (!code.trim()) {
      alert('Please enter some code');
      return;
    }

    try {
      setLoading(true);
      const res = await API.post('/compiler/execute', {
        code,
        language
      }).catch(() => ({
        data: { output: 'Compiler not available', errors: [] }
      }));

      setOutput(res.data.output || 'Execution complete');
    } catch (error) {
      console.error('Error compiling:', error);
      setOutput('Error executing code');
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
            Code Compiler
          </h1>
          <p className="font-mono text-[11px] text-white/40 tracking-wider uppercase">
            Write & Execute Code
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Editor */}
          <motion.div
            variants={fadeUp}
            className="border border-white/[.06] p-6"
          >
            <div className="flex justify-between items-center mb-4">
              <h2 className="font-display text-lg font-bold text-white uppercase">Editor</h2>
              <select
                value={language}
                onChange={(e) => setLanguage(e.target.value)}
                className="bg-white/5 border border-white/[.06] px-3 py-2 text-white font-mono text-[10px] focus:border-[#B7FF4A] outline-none"
              >
                <option value="python">Python</option>
                <option value="javascript">JavaScript</option>
                <option value="java">Java</option>
                <option value="cpp">C++</option>
              </select>
            </div>

            <textarea
              value={code}
              onChange={(e) => setCode(e.target.value)}
              placeholder="Write your code here..."
              className="w-full h-96 bg-white/5 border border-white/[.06] p-4 text-white font-mono text-[11px] focus:border-[#B7FF4A] outline-none resize-none"
            />

            <button
              onClick={handleCompile}
              disabled={loading}
              className="w-full mt-4 px-6 py-3 bg-[#B7FF4A] text-black font-bold uppercase tracking-wider disabled:opacity-50 hover:bg-[#B7FF4A]/90 transition-all duration-300"
            >
              {loading ? 'Running...' : 'Run Code'}
            </button>
          </motion.div>

          {/* Output */}
          <motion.div
            variants={fadeUp}
            className="border border-white/[.06] p-6"
          >
            <h2 className="font-display text-lg font-bold text-white mb-4 uppercase">Output</h2>
            <div className="h-96 bg-white/5 border border-white/[.06] p-4 rounded overflow-y-auto">
              <p className="font-mono text-[11px] text-white/60 whitespace-pre-wrap">
                {output || 'Output will appear here...'}
              </p>
            </div>
          </motion.div>
        </div>
      </motion.div>
    </DashboardLayout>
  );
}
