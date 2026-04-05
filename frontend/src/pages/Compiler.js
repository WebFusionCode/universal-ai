import React from 'react';
import DashboardLayout from '../components/DashboardLayout';
import Editor from "@monaco-editor/react";
import { motion } from 'framer-motion';

const fadeUp = { hidden: { opacity: 0, y: 20 }, visible: { opacity: 1, y: 0 } };

export default function Compiler() {
  return (
    <DashboardLayout title="Compiler">
      <motion.div initial="hidden" animate="visible" variants={{ visible: { transition: { staggerChildren: 0.06 } } }}>
        <motion.div variants={fadeUp} className="mb-8">
          <h1 className="font-display text-2xl font-bold uppercase tracking-tight text-[#B7FF4A]">ML Compiler</h1>
          <p className="font-mono text-[11px] text-white/30 tracking-wider uppercase mt-1">Write, execute, and analyze ML pipelines instantly</p>
        </motion.div>
        
        <motion.div variants={fadeUp} className="border border-white/[.06] overflow-hidden">
          <Editor
            height="75vh"
            defaultLanguage="python"
            defaultValue="# Write your ML code here
# Example:
# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier

def train_model(data):
    print('Training model...')
    # Your logic
    return True
"
            theme="vs-dark"
            options={{
              minimap: { enabled: false },
              fontSize: 14,
              fontFamily: '"JetBrains Mono", monospace',
              scrollBeyondLastLine: false,
              padding: { top: 20 }
            }}
          />
        </motion.div>
      </motion.div>
    </DashboardLayout>
  );
}
