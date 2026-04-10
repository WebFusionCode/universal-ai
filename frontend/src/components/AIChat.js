import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import API from '../lib/api';

export default function AIChat() {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState([
    { role: 'assistant', text: 'I\'m your AI ML advisor. Ask me about datasets, models, training strategies, or debugging.' }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef(null);

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim() || loading) return;
    const userMsg = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', text: userMsg }]);
    setLoading(true);
    try {
      const res = await API.post('/chat', { message: userMsg, dataset_info: '' });
      setMessages(prev => [...prev, { role: 'assistant', text: res.data.response || 'No response.' }]);
    } catch (err) {
      setMessages(prev => [...prev, { role: 'assistant', text: 'Failed to respond. Try again.' }]);
    } finally { setLoading(false); }
  };

  return (
    <>
      <motion.button data-testid="ai-chat-toggle" onClick={() => setOpen(!open)}
        className="fixed bottom-6 right-6 z-50 w-12 h-12 border border-white/[.12] bg-[#111] text-[#B7FF4A] flex items-center justify-center hover:border-[#B7FF4A]/30 transition-all"
        whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
        {open ? (
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M18 6L6 18M6 6l12 12"/></svg>
        ) : (
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/></svg>
        )}
      </motion.button>

      <AnimatePresence>
        {open && (
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 20 }}
            className="fixed bottom-20 right-6 z-50 w-80 h-[460px] flex flex-col border border-white/[.08] bg-[#0e0e0e] shadow-2xl shadow-black/50">
            <div className="flex items-center gap-3 px-5 py-3 border-b border-white/[.06]">
              <div className="w-2 h-2 bg-[#B7FF4A] rounded-full" />
              <p className="font-mono text-[10px] tracking-[0.15em] uppercase text-white/50">AI Assistant</p>
            </div>

            <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 space-y-3">
              {messages.map((msg, i) => (
                <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div className={`max-w-[85%] px-3 py-2 font-mono text-[11px] leading-relaxed ${
                    msg.role === 'user'
                      ? 'bg-[#B7FF4A]/10 text-[#B7FF4A] border border-[#B7FF4A]/15'
                      : 'bg-white/[.04] text-white/60 border border-white/[.06]'
                  }`}>{msg.text}</div>
                </div>
              ))}
              {loading && (
                <div className="flex justify-start">
                  <div className="bg-white/[.04] border border-white/[.06] text-white/30 px-3 py-2 font-mono text-[11px] animate-pulse">Thinking...</div>
                </div>
              )}
            </div>

            <div className="p-3 border-t border-white/[.06]">
              <div className="flex items-center gap-2">
                <input data-testid="ai-chat-input" value={input} onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), sendMessage())}
                  placeholder="Ask about ML..."
                  className="flex-1 bg-white/[.03] border border-white/[.06] px-3 py-2 font-mono text-[11px] text-white placeholder-white/20 outline-none focus:border-[#B7FF4A]/30" />
                <button data-testid="ai-chat-send" onClick={sendMessage} disabled={!input.trim() || loading}
                  className="w-8 h-8 bg-[#B7FF4A] flex items-center justify-center text-[#0a0a0a] disabled:opacity-20 hover:bg-[#c8ff73] transition-all">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"/></svg>
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
