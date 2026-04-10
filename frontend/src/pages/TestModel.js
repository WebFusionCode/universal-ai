import React, { useState } from 'react';
import DashboardLayout from '../components/DashboardLayout';
import { Upload, Zap, CheckCircle2, AlertCircle, FileText, ImageIcon } from 'lucide-react';
import API from '../lib/api';

export default function TestModel() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [isImage, setIsImage] = useState(false);

  const handleFileSelect = (e) => {
    const selectedFile = e.target.files[0];
    if (!selectedFile) return;
    
    setFile(selectedFile);
    setIsImage(selectedFile.type.startsWith('image/'));
    setError('');
    setPrediction(null);
  };

  const handleTest = async () => {
    if (!file) {
      setError('Please upload a file or image first');
      return;
    }

    setLoading(true);
    setError('');
    
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      const endpoint = isImage ? '/test-image' : '/test-model';
      const res = await API.post(endpoint, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      if (res.data.error) {
        setError(res.data.error);
      } else {
        setPrediction(res.data);
      }
    } catch (err) {
      setError(err.response?.data?.error || 'Inference failed. Ensure a model is trained.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <DashboardLayout>
      <div className="p-8">
        <header className="mb-8">
          <h1 className="text-3xl font-bold text-white mb-2">Test Your Model</h1>
          <p className="text-slate-400">Upload a fresh sample to see your AI in action</p>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Upload Section */}
          <div className="bg-slate-900/50 border border-slate-800 rounded-3xl p-8 backdrop-blur-xl">
            <div className="mb-6">
              <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
                <Upload className="w-5 h-5 text-indigo-400" />
                Upload Sample
              </h2>
              
              <div className="relative group">
                <input
                  type="file"
                  onChange={handleFileSelect}
                  className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                />
                <div className={`border-2 border-dashed rounded-2xl p-12 text-center transition-all ${
                  file ? 'border-indigo-500 bg-indigo-500/5' : 'border-slate-800 group-hover:border-slate-700 bg-slate-950/50'
                }`}>
                  {isImage && file ? (
                    <img src={URL.createObjectURL(file)} alt="Preview" className="w-32 h-32 mx-auto rounded-xl object-cover mb-4" />
                  ) : (
                    <div className="w-16 h-16 bg-slate-900 rounded-2xl flex items-center justify-center mx-auto mb-4">
                      {isImage ? <ImageIcon className="w-8 h-8 text-slate-400" /> : <FileText className="w-8 h-8 text-slate-400" />}
                    </div>
                  )}
                  <p className="text-white font-medium mb-1">
                    {file ? file.name : 'Click to upload or drag & drop'}
                  </p>
                  <p className="text-slate-500 text-sm">CSV, Excel, or Images</p>
                </div>
              </div>
            </div>

            <button
              onClick={handleTest}
              disabled={loading || !file}
              className={`w-full py-4 rounded-xl font-semibold flex items-center justify-center gap-2 transition-all ${
                loading || !file 
                  ? 'bg-slate-800 text-slate-500 cursor-not-allowed' 
                  : 'bg-indigo-600 hover:bg-indigo-500 text-white shadow-lg shadow-indigo-500/20'
              }`}
            >
              {loading ? (
                <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              ) : (
                <>
                  <Zap className="w-5 h-5" />
                  Run Inference
                </>
              )}
            </button>

            {error && (
              <div className="mt-4 p-4 bg-red-500/10 border border-red-500/20 rounded-xl flex items-start gap-3">
                <AlertCircle className="w-5 h-5 text-red-500 shrink-0 mt-0.5" />
                <p className="text-red-200 text-sm">{error}</p>
              </div>
            )}
          </div>

          {/* Results Section */}
          <div className="bg-slate-900/50 border border-slate-800 rounded-3xl p-8 backdrop-blur-xl">
            <h2 className="text-xl font-semibold text-white mb-6 flex items-center gap-2">
              <CheckCircle2 className="w-5 h-5 text-emerald-400" />
              Intelligence Output
            </h2>

            {prediction ? (
              <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
                <div className="p-6 bg-emerald-500/10 border border-emerald-500/20 rounded-2xl">
                  <p className="text-emerald-400 text-sm font-medium uppercase tracking-wider mb-2">Primary Prediction</p>
                  <p className="text-4xl font-bold text-white tracking-tight">
                    {isImage ? prediction.prediction : (Array.isArray(prediction.predictions) ? prediction.predictions[0] : 'N/A')}
                  </p>
                </div>

                {!isImage && prediction.predictions && (
                  <div className="border border-slate-800 rounded-2xl overflow-hidden">
                    <div className="bg-slate-950 p-4 border-b border-slate-800">
                      <p className="text-white font-medium">Batch Predictions ({prediction.count} rows)</p>
                    </div>
                    <div className="p-4 max-h-[300px] overflow-y-auto font-mono text-sm">
                      {prediction.predictions.slice(0, 10).map((p, idx) => (
                        <div key={idx} className="flex justify-between py-2 border-b border-slate-800/50 last:border-0">
                          <span className="text-slate-500">Row {idx + 1}</span>
                          <span className="text-white font-bold">{p}</span>
                        </div>
                      ))}
                      {prediction.count > 10 && (
                        <p className="text-center text-slate-500 pt-4 italic">And {prediction.count - 10} more rows...</p>
                      )}
                    </div>
                  </div>
                )}

                <div className="flex items-center gap-3 p-4 bg-indigo-500/5 border border-indigo-500/10 rounded-xl">
                  <div className="w-2 h-2 rounded-full bg-indigo-400 animate-pulse" />
                  <p className="text-indigo-200 text-sm italic underline decoration-indigo-400/30 underline-offset-4 font-normal">
                    AI engine verified predictions based on your optimized model.
                  </p>
                </div>
              </div>
            ) : (
              <div className="h-full flex flex-col items-center justify-center text-center opacity-40 py-20">
                <div className="w-20 h-20 bg-slate-950 border border-slate-800 rounded-Full flex items-center justify-center mb-6">
                  <Zap className="w-10 h-10 text-slate-600" />
                </div>
                <p className="text-slate-400 font-medium">No results to display</p>
                <p className="text-slate-600 text-sm max-w-[200px] mt-2">Results will appear here once you run inference.</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
}
