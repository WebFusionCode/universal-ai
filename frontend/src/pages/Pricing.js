import React from 'react';
import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';

const fadeUp = { hidden: { opacity: 0, y: 20 }, visible: { opacity: 1, y: 0 } };

export default function Pricing() {
  const plans = [
    {
      name: 'Starter',
      price: '$29',
      period: '/month',
      features: [
        'Up to 10 experiments',
        'Basic model training',
        'Community support',
        'Standard compute',
        '1 team member'
      ],
      color: '#B7FF4A'
    },
    {
      name: 'Professional',
      price: '$99',
      period: '/month',
      features: [
        'Unlimited experiments',
        'Advanced model training',
        'Email support',
        'GPU compute',
        'Up to 5 team members',
        'Model deployment'
      ],
      color: '#6AA7FF',
      popular: true
    },
    {
      name: 'Enterprise',
      price: 'Custom',
      period: '',
      features: [
        'Everything in Pro',
        'Dedicated support',
        'Custom infrastructure',
        'Unlimited team members',
        'API access',
        'Priority queue'
      ],
      color: '#FF6B9D'
    }
  ];

  return (
    <div className="relative min-h-screen bg-[#0a0a0a] overflow-hidden pt-32 pb-20">
      {/* Background orbs */}
      <div className="absolute top-0 left-0 w-[600px] h-[600px] rounded-full bg-[#B7FF4A]/5 blur-[150px]" />
      <div className="absolute bottom-0 right-0 w-[500px] h-[500px] rounded-full bg-[#6AA7FF]/5 blur-[130px]" />

      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="max-w-6xl mx-auto px-4 mb-16 relative z-10 text-center"
      >
        <h1 className="font-display text-5xl font-bold uppercase tracking-tight text-white mb-4">
          Simple, Transparent Pricing
        </h1>
        <p className="font-mono text-[12px] text-white/40 tracking-widest uppercase">
          Choose the perfect plan for your AI journey
        </p>
      </motion.div>

      {/* Pricing Cards */}
      <div className="max-w-6xl mx-auto px-4 grid grid-cols-1 md:grid-cols-3 gap-6 relative z-10">
        {plans.map((plan, idx) => (
          <motion.div
            key={idx}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: idx * 0.1 }}
            className={`border p-8 transition-all duration-300 ${
              plan.popular
                ? 'border-[#B7FF4A] bg-[#B7FF4A]/5 relative z-20 scale-105 origin-center'
                : 'border-white/[.06] hover:border-white/[.12]'
            }`}
          >
            {plan.popular && (
              <div className="absolute -top-4 left-1/2 -translate-x-1/2">
                <span
                  className="px-4 py-1 font-mono text-[10px] font-bold uppercase tracking-widest text-white"
                  style={{ backgroundColor: plan.color, color: '#000' }}
                >
                  Most Popular
                </span>
              </div>
            )}

            <h3 className="font-display text-2xl font-bold uppercase text-white mb-2">
              {plan.name}
            </h3>
            <div className="mb-6">
              <span className="font-display text-5xl font-bold" style={{ color: plan.color }}>
                {plan.price}
              </span>
              <span className="font-mono text-[11px] text-white/40 ml-2">{plan.period}</span>
            </div>

            <button
              className="w-full px-6 py-3 mb-8 font-bold uppercase tracking-wider transition-all duration-300"
              style={{
                backgroundColor: plan.color,
                color: '#000'
              }}
              onMouseEnter={(e) => {
                e.target.style.opacity = '0.8';
              }}
              onMouseLeave={(e) => {
                e.target.style.opacity = '1';
              }}
            >
              Get Started
            </button>

            <ul className="space-y-3">
              {plan.features.map((feature, fIdx) => (
                <li key={fIdx} className="flex items-start gap-3">
                  <span style={{ color: plan.color }}>✓</span>
                  <span className="font-mono text-[11px] text-white/60">{feature}</span>
                </li>
              ))}
            </ul>
          </motion.div>
        ))}
      </div>

      {/* Back to Dashboard */}
      <div className="max-w-6xl mx-auto px-4 mt-16 text-center relative z-10">
        <Link
          to="/dashboard"
          className="inline-block font-mono text-[11px] uppercase tracking-widest text-white/60 hover:text-white transition"
        >
          ← Back to Dashboard
        </Link>
      </div>
    </div>
  );
}
