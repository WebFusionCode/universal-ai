import React, { useState, useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import DashboardLayout from '../components/DashboardLayout';
import API from '../lib/api';

const fadeUp = { hidden: { opacity: 0, y: 20 }, visible: { opacity: 1, y: 0 } };

export default function Chatbot() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage = { role: 'user', content: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const res = await API.post('/chat', {
        message: input,
        history: messages
      }).catch(() => ({
        data: { response: 'Chat service is not available. Please try again later.' }
      }));

      const botMessage = {
        role: 'assistant',
        content: res.data.response || 'No response available'
      };
      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage = {
        role: 'assistant',
        content: 'Sorry, there was an error processing your message.'
      };
      setMessages((prev) => [...prev, errorMessage]);
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
        className="h-[calc(100vh-200px)] flex flex-col"
      >
        <div>
          <h1 className="font-display text-4xl font-bold uppercase tracking-tight text-white mb-2">
            AI Chatbot
          </h1>
          <p className="font-mono text-[11px] text-white/40 tracking-wider uppercase">
            Ask Me Anything About Your Models
          </p>
        </div>

        <div className="flex-1 mt-8 flex flex-col border border-white/[.06]">
          {/* Chat Messages */}
          <div className="flex-1 overflow-y-auto p-6 space-y-4">
            {messages.length === 0 ? (
              <div className="flex items-center justify-center h-full">
                <p className="font-mono text-[12px] text-white/30 text-center">
                  Start a conversation by typing a message below...
                </p>
              </div>
            ) : (
              messages.map((msg, idx) => (
                <motion.div
                  key={idx}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={`flex ${
                    msg.role === 'user' ? 'justify-end' : 'justify-start'
                  }`}
                >
                  <div
                    className={`max-w-md px-4 py-3 rounded ${
                      msg.role === 'user'
                        ? 'bg-[#B7FF4A]/10 border border-[#B7FF4A]/30 text-white'
                        : 'bg-white/5 border border-white/[.06] text-white/80'
                    }`}
                  >
                    <p className="font-mono text-[11px] leading-relaxed">
                      {msg.content}
                    </p>
                  </div>
                </motion.div>
              ))
            )}

            {loading && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex justify-start"
              >
                <div className="flex gap-2 px-4 py-3">
                  <div className="w-2 h-2 bg-[#B7FF4A] rounded-full animate-bounce" />
                  <div
                    className="w-2 h-2 bg-[#B7FF4A] rounded-full animate-bounce"
                    style={{ animationDelay: '0.1s' }}
                  />
                  <div
                    className="w-2 h-2 bg-[#B7FF4A] rounded-full animate-bounce"
                    style={{ animationDelay: '0.2s' }}
                  />
                </div>
              </motion.div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Input Section */}
          <form
            onSubmit={handleSendMessage}
            className="border-t border-white/[.06] p-4 flex gap-4"
          >
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Type your message..."
              disabled={loading}
              className="flex-1 bg-white/5 border border-white/[.06] px-4 py-2 text-white font-mono text-[11px] placeholder-white/30 focus:border-[#B7FF4A] outline-none transition disabled:opacity-50"
            />
            <button
              type="submit"
              disabled={loading || !input.trim()}
              className="px-6 py-2 bg-[#B7FF4A] text-black font-bold uppercase tracking-wider disabled:opacity-50 hover:bg-[#B7FF4A]/90 transition-all duration-300"
            >
              Send
            </button>
          </form>
        </div>
      </motion.div>
    </DashboardLayout>
  );
}
