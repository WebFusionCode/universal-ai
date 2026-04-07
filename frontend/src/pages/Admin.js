import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import DashboardLayout from '../components/DashboardLayout';
import API from '../lib/api';

const fadeUp = { hidden: { opacity: 0, y: 20 }, visible: { opacity: 1, y: 0 } };

export default function Admin() {
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState({
    totalUsers: 0,
    totalExperiments: 0,
    totalModels: 0,
    systemHealth: '100%'
  });

  useEffect(() => {
    loadAdminData();
  }, []);

  const loadAdminData = async () => {
    try {
      setLoading(true);
      const usersRes = await API.get('/admin/users').catch(() => ({ data: { users: [] } }));
      const statsRes = await API.get('/admin/stats').catch(() => ({ data: stats }));

      setUsers(usersRes.data.users || []);
      setStats(statsRes.data || stats);
    } catch (error) {
      console.error('Error loading admin data:', error);
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
            Admin Dashboard
          </h1>
          <p className="font-mono text-[11px] text-white/40 tracking-wider uppercase">
            System Management & User Control
          </p>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {[
            { label: 'Total Users', value: stats.totalUsers, color: '#B7FF4A' },
            { label: 'Total Experiments', value: stats.totalExperiments, color: '#6AA7FF' },
            { label: 'Total Models', value: stats.totalModels, color: '#FF6B9D' },
            { label: 'System Health', value: stats.systemHealth, color: '#4FD1C5' }
          ].map((stat, idx) => (
            <motion.div
              key={idx}
              variants={fadeUp}
              className="border border-white/[.06] p-6 hover:border-white/[.12] transition-all duration-300"
            >
              <p className="font-mono text-[10px] tracking-[0.15em] uppercase text-white/30 mb-2">
                {stat.label}
              </p>
              <p className="font-display text-2xl font-bold" style={{ color: stat.color }}>
                {stat.value}
              </p>
            </motion.div>
          ))}
        </div>

        {/* Users Table */}
        <motion.div
          variants={fadeUp}
          className="border border-white/[.06] p-6 overflow-x-auto"
        >
          <h2 className="font-display text-lg font-bold text-white mb-4 uppercase">Users</h2>
          <table className="w-full">
            <thead>
              <tr className="border-b border-white/[.06]">
                <th className="px-4 py-3 text-left font-mono text-[10px] text-white/30 uppercase">ID</th>
                <th className="px-4 py-3 text-left font-mono text-[10px] text-white/30 uppercase">Email</th>
                <th className="px-4 py-3 text-left font-mono text-[10px] text-white/30 uppercase">Role</th>
                <th className="px-4 py-3 text-left font-mono text-[10px] text-white/30 uppercase">Status</th>
              </tr>
            </thead>
            <tbody>
              {users.map((user, idx) => (
                <tr key={idx} className="border-b border-white/[.06] hover:bg-white/[.02] transition">
                  <td className="px-4 py-3 font-mono text-[11px] text-white/60">{user.id}</td>
                  <td className="px-4 py-3 font-mono text-[11px] text-white/60">{user.email}</td>
                  <td className="px-4 py-3 font-mono text-[11px] text-white/60">{user.role || 'user'}</td>
                  <td className="px-4 py-3 font-mono text-[11px]">
                    <span className="text-[#B7FF4A]">Active</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </motion.div>
      </motion.div>
    </DashboardLayout>
  );
}
