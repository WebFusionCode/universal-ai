import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import DashboardLayout from '../components/DashboardLayout';
import { Upload, FileText, Target, Zap, CheckCircle2, AlertCircle } from 'lucide-react';
import API from '../lib/api';

const safeNumber = (val) =>
  typeof val === 'number' ? val.toFixed(4) : 'N/A';

export default function Train() {
  const navigate = useNavigate();
  const [step, setStep] = useState(1);
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [targetColumn, setTargetColumn] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [trainingComplete, setTrainingComplete] = useState(false);
  const [result, setResult] = useState(null);
  const [bestModel, setBestModel] = useState(null);
  const [score, setScore] = useState(null);

  const handleFileSelect = async (e) => {
    const selectedFile = e.target.files[0];
    if (!selectedFile) return;

    const isZip = selectedFile.name.endsWith('.zip');
    setFile(selectedFile);
    setError('');
    setLoading(true);

    try {
      if (isZip) {
        // For ZIP files (images), skip preview and go to step 2
        setPreview({ columns: [], preview: [], is_image: true });
        setStep(2);
      } else {
        // For CSV/Excel files, get preview
        const formData = new FormData();
        formData.append('file', selectedFile);

        const res = await API.post('/preview', formData, {
          headers: { 'Content-Type': 'multipart/form-data' }
        });

        setPreview(res.data || { columns: [], preview: [] });
        setStep(2);
      }
    } catch (err) {
      setError(err.response?.data?.detail || err.response?.data?.error || 'Failed to preview dataset');
    } finally {
      setLoading(false);
    }
  };

  const handleTrain = async () => {
    const isZip = file?.name.endsWith('.zip');

    if (!isZip && !targetColumn) {
      setError('Please select a target column');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const formData = new FormData();
      formData.append('file', file);
      if (!isZip) {
        formData.append('target_column', targetColumn);
      }
      formData.append('dataset_type', isZip ? 'image' : 'tabular');

      const res = await API.post('/train', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      const data = res.data;
      console.log("TRAIN RESPONSE:", data);

      setResult(data);
      setBestModel(data?.best_model ?? null);
      setScore(data?.score ?? null);
      setTrainingComplete(true);
      setStep(3);
    } catch (err) {
      setError(err.response?.data?.detail || err.response?.data?.error || 'Training failed');
    } finally {
      setLoading(false);
    }
  };

  const resetForm = () => {
    setStep(1);
    setFile(null);
    setPreview(null);
    setTargetColumn('');
    setTrainingComplete(false);
    setResult(null);
    setBestModel(null);
    setScore(null);
    setError('');
  };

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="font-display text-2xl font-bold uppercase tracking-tight text-white">Train Model</h1>
            <p className="font-mono text-[11px] text-white/40 tracking-wider uppercase mt-1">
              Upload dataset and train ML models automatically
            </p>
          </div>
        </div>

        {/* Progress Steps */}
        <div className="flex items-center gap-4">
          {[
            { num: 1, label: 'Upload', icon: Upload },
            { num: 2, label: 'Configure', icon: Target },
            { num: 3, label: 'Results', icon: CheckCircle2 }
          ].map(({ num, label, icon: Icon }) => (
            <div key={num} className="flex items-center gap-2">
              <div className={`flex items-center justify-center w-8 h-8 rounded-full border ${
                step >= num 
                  ? 'bg-[#B7FF4A] border-[#B7FF4A] text-[#0a0a0a]' 
                  : 'bg-transparent border-white/20 text-white/40'
              }`}>
                <span className="font-mono text-xs font-bold">{num}</span>
              </div>
              <span className={`font-mono text-[10px] uppercase tracking-wider ${
                step >= num ? 'text-white' : 'text-white/40'
              }`}>{label}</span>
              {num < 3 && <div className={`h-[1px] w-8 ${step > num ? 'bg-[#B7FF4A]' : 'bg-white/20'}`} />}
            </div>
          ))}
        </div>

        {/* Error Display */}
        {error && (
          <div className="border border-[#FF5C7A]/20 bg-[#FF5C7A]/5 px-4 py-3 flex items-start gap-3">
            <AlertCircle size={16} className="text-[#FF5C7A] mt-0.5" />
            <span className="font-mono text-[11px] text-[#FF5C7A]">{error}</span>
          </div>
        )}

        {/* Step 1: Upload */}
        {step === 1 && (
          <div className="border border-white/[.08] bg-[#111] p-8">
            <div className="text-center">
              <Upload size={48} className="text-[#B7FF4A] mx-auto mb-4" />
              <h3 className="font-display text-lg font-bold uppercase text-white mb-2">Upload Dataset</h3>
              <p className="font-mono text-[11px] text-white/40 mb-6">
                CSV, Excel, or ZIP (for images) supported
              </p>
              <label className="cursor-pointer inline-block">
                <div className="px-6 py-3 bg-[#B7FF4A] text-[#0a0a0a] font-mono text-[11px] font-bold tracking-[0.1em] uppercase hover:bg-[#c8ff73] transition-all">
                  Select File
                </div>
                <input
                  type="file"
                  accept=".csv,.xlsx,.zip"
                  onChange={handleFileSelect}
                  className="hidden"
                  disabled={loading}
                />
              </label>
              {loading && (
                <p className="font-mono text-[11px] text-[#B7FF4A] mt-4 animate-pulse">
                  Processing file...
                </p>
              )}
            </div>
          </div>
        )}

        {/* Step 2: Configure */}
        {step === 2 && preview && (
          <div className="space-y-6">
            {/* Dataset Preview */}
            <div className="border border-white/[.08] bg-[#111] p-6">
              <h3 className="font-display text-sm font-bold uppercase text-white mb-4 flex items-center gap-2">
                <FileText size={16} className="text-[#B7FF4A]" />
                Dataset Preview
              </h3>
              
              <div className="grid grid-cols-3 gap-4 mb-6">
                <div>
                  <div className="font-mono text-[10px] text-white/40 uppercase mb-1">Rows</div>
                  <div className="font-mono text-xl text-white font-bold">{preview.rows || 0}</div>
                </div>
                <div>
                  <div className="font-mono text-[10px] text-white/40 uppercase mb-1">Columns</div>
                  <div className="font-mono text-xl text-white font-bold">{preview.columns?.length || 0}</div>
                </div>
                <div>
                  <div className="font-mono text-[10px] text-white/40 uppercase mb-1">Type</div>
                  <div className="font-mono text-xl text-white font-bold capitalize">{preview.problem_type || 'Auto'}</div>
                </div>
              </div>

              {/* Preview Table */}
              {preview.preview && preview.preview.length > 0 && (
                <div className="overflow-x-auto">
                  <table className="w-full border border-white/[.08]">
                    <thead>
                      <tr className="bg-white/[.03]">
                        {Object.keys(preview.preview[0]).map((col) => (
                          <th key={col} className="px-4 py-2 text-left font-mono text-[10px] text-white/60 uppercase">
                            {col}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {preview.preview.map((row, idx) => (
                        <tr key={idx} className="border-t border-white/[.08]">
                          {Object.values(row).map((val, i) => (
                            <td key={i} className="px-4 py-2 font-mono text-[11px] text-white/80">
                              {String(val)}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>

            {/* Target Column Selection */}
            <div className="border border-white/[.08] bg-[#111] p-6">
              <h3 className="font-display text-sm font-bold uppercase text-white mb-4 flex items-center gap-2">
                <Target size={16} className="text-[#B7FF4A]" />
                Select Target Column
              </h3>
              
              {!preview?.is_image && (
                <select
                  value={targetColumn}
                  onChange={(e) => setTargetColumn(e.target.value)}
                  className="w-full px-4 py-3 bg-white/[.03] border border-white/[.08] text-white font-mono text-[13px] focus:outline-none focus:border-[#B7FF4A]/40 transition-all"
                  data-testid="target-column-select"
                >
                  <option value="">Select target column...</option>
                  {Array.isArray(preview?.columns) && preview.columns.map((col) => (
                    <option key={col} value={col}>{col}</option>
                  ))}
                </select>
              )}
              {preview?.is_image && (
                <div className="px-4 py-3 bg-white/[.03] border border-white/[.08] text-white font-mono text-[13px]">
                  📁 Image dataset detected - ready to train CNN
                </div>
              )}

              <div className="flex gap-3 mt-6">
                <button
                  onClick={handleTrain}
                  disabled={!file || loading}
                  className="px-6 py-3 bg-[#B7FF4A] text-[#0a0a0a] font-mono text-[11px] font-bold tracking-[0.1em] uppercase hover:bg-[#c8ff73] transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                  data-testid="train-button"
                >
                  <Zap size={14} />
                  {loading ? 'Training...' : 'Train Model'}
                </button>
                <button
                  onClick={resetForm}
                  disabled={loading}
                  className="px-6 py-3 border border-white/[.08] text-white font-mono text-[11px] font-bold tracking-[0.1em] uppercase hover:bg-white/[.03] transition-all disabled:opacity-50"
                >
                  Reset
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Step 3: Results */}
        {step === 3 && result && (
          <div className="border border-white/[.08] bg-[#111] p-6">
            <div className="text-center mb-6">
              <CheckCircle2 size={48} className="text-[#B7FF4A] mx-auto mb-4" />
              <h3 className="font-display text-lg font-bold uppercase text-white mb-2">Training Complete!</h3>
            </div>

            {/* Results Grid */}
            <div className="grid grid-cols-2 gap-4 mb-6">
              <div className="border border-white/[.08] bg-white/[.02] p-4">
                <div className="font-mono text-[10px] text-white/40 uppercase mb-1">Best Model</div>
                <div className="font-mono text-xl text-[#B7FF4A] font-bold">{bestModel || "N/A"}</div>
              </div>
              <div className="border border-white/[.08] bg-white/[.02] p-4">
                <div className="font-mono text-[10px] text-white/40 uppercase mb-1">Score</div>
                <div className="font-mono text-xl text-[#B7FF4A] font-bold">
                  {safeNumber(score)}
                </div>
              </div>
            </div>

            <div className="flex gap-3">
              <button
                onClick={() => navigate('/experiments')}
                className="px-6 py-3 bg-[#B7FF4A] text-[#0a0a0a] font-mono text-[11px] font-bold tracking-[0.1em] uppercase hover:bg-[#c8ff73] transition-all"
              >
                View Experiments
              </button>
              <button
                onClick={resetForm}
                className="px-6 py-3 border border-white/[.08] text-white font-mono text-[11px] font-bold tracking-[0.1em] uppercase hover:bg-white/[.03] transition-all"
              >
                Train Another
              </button>
            </div>
          </div>
        )}
      </div>
    </DashboardLayout>
  );
}
