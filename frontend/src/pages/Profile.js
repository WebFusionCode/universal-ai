import React, { useState, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import DashboardLayout from '../components/DashboardLayout';
import API from '../lib/api';

const fadeUp = { hidden: { opacity: 0, y: 20 }, visible: { opacity: 1, y: 0 } };

export default function Profile() {
  const userId = localStorage.getItem('user_id');
  const email = localStorage.getItem('email') || '';
  const [profile, setProfile] = useState({ name: '', phone: '', dob: '' });
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  const load = useCallback(async () => {
    if (!userId) return;
    try {
      const res = await API.get(`/api/profile/${userId}`);
      setProfile({ name: res.data.name || '', phone: res.data.phone || '', dob: res.data.dob || '' });
    } catch (e) {}
  }, [userId]);

  useEffect(() => { load(); }, [load]);

  const handleSave = async () => {
    setSaving(true); setSaved(false);
    try {
      await API.post('/api/update-profile', { user_id: userId, ...profile });
      setSaved(true);
      setTimeout(() => setSaved(false), 3000);
    } catch (e) {} finally { setSaving(false); }
  };

  return (
    <DashboardLayout title="Profile">
      <motion.div initial="hidden" animate="visible" variants={{ visible: { transition: { staggerChildren: 0.06 } } }}>
        <motion.div variants={fadeUp} className="mb-8">
          <h1 className="font-display text-2xl font-bold uppercase tracking-tight">Profile</h1>
          <p className="font-mono text-[11px] text-white/30 tracking-wider uppercase mt-1">Manage your account</p>
        </motion.div>

        <motion.div variants={fadeUp} className="max-w-md">
          {/* Avatar */}
          <div className="flex items-center gap-5 mb-8">
            <div className="w-14 h-14 border border-white/[.08] flex items-center justify-center font-display text-xl font-bold text-[#B7FF4A]">
              {(profile.name || email || '?')[0].toUpperCase()}
            </div>
            <div>
              <p className="font-display text-base font-bold text-white">{profile.name || 'Anonymous'}</p>
              <p className="font-mono text-[10px] text-white/30 tracking-wider">{email}</p>
            </div>
          </div>

          <div className="space-y-5">
            {[
              { id: 'profile-name', label: 'Name', key: 'name', type: 'text', placeholder: 'Your name' },
              { id: 'profile-phone', label: 'Phone', key: 'phone', type: 'text', placeholder: '+1 234 567 890' },
              { id: 'profile-dob', label: 'Date of Birth', key: 'dob', type: 'date', placeholder: '' },
            ].map((field) => (
              <div key={field.key}>
                <label className="font-mono text-[10px] text-white/30 tracking-[0.15em] uppercase mb-1.5 block">{field.label}</label>
                <input data-testid={field.id} type={field.type} value={profile[field.key]}
                  onChange={(e) => setProfile(p => ({ ...p, [field.key]: e.target.value }))}
                  className="w-full px-4 py-3 bg-white/[.03] border border-white/[.08] text-white font-mono text-[13px] placeholder-white/15 focus:outline-none focus:border-[#B7FF4A]/40 transition-all"
                  placeholder={field.placeholder} />
              </div>
            ))}
            <button data-testid="profile-save" onClick={handleSave} disabled={saving}
              className="w-full py-3 bg-[#B7FF4A] text-[#0a0a0a] font-mono text-[11px] font-bold tracking-[0.1em] uppercase hover:bg-[#c8ff73] transition-all disabled:opacity-50">
              {saving ? 'Saving...' : saved ? 'Saved!' : 'Save Changes'}
            </button>
          </div>
        </motion.div>
      </motion.div>
    </DashboardLayout>
  );
}
