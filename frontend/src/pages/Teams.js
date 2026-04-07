import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import DashboardLayout from '../components/DashboardLayout';
import API from '../lib/api';

const fadeUp = { hidden: { opacity: 0, y: 20 }, visible: { opacity: 1, y: 0 } };

export default function Teams() {
  const [teams, setTeams] = useState([]);
  const [members, setMembers] = useState([]);
  const [selectedTeam, setSelectedTeam] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadTeams();
  }, []);

  const loadTeams = async () => {
    try {
      setLoading(true);
      const res = await API.get('/teams').catch(() => ({ data: { teams: [] } }));
      setTeams(res.data.teams || []);
    } catch (error) {
      console.error('Error loading teams:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSelectTeam = async (team) => {
    setSelectedTeam(team);
    try {
      const res = await API.get(`/teams/${team.id}/members`).catch(() => ({
        data: { members: [] }
      }));
      setMembers(res.data.members || []);
    } catch (error) {
      console.error('Error loading team members:', error);
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
            Teams
          </h1>
          <p className="font-mono text-[11px] text-white/40 tracking-wider uppercase">
            Collaborate With Your Team
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Teams List */}
          <motion.div
            variants={fadeUp}
            className="border border-white/[.06] p-6"
          >
            <h2 className="font-display text-lg font-bold text-white mb-4 uppercase">Your Teams</h2>

            {teams.length === 0 ? (
              <p className="font-mono text-[11px] text-white/30">No teams yet</p>
            ) : (
              <div className="space-y-2">
                {teams.map((team, idx) => (
                  <motion.button
                    key={idx}
                    whileHover={{ paddingLeft: '1rem' }}
                    onClick={() => handleSelectTeam(team)}
                    className={`w-full text-left px-3 py-2 border border-transparent transition-all duration-300 ${
                      selectedTeam?.id === team.id
                        ? 'border-[#B7FF4A] bg-[#B7FF4A]/5 text-[#B7FF4A]'
                        : 'border-white/[.06] text-white/60 hover:border-white/[.12]'
                    }`}
                  >
                    <p className="font-mono text-[11px] font-bold uppercase truncate">
                      {team.name || 'Untitled Team'}
                    </p>
                  </motion.button>
                ))}
              </div>
            )}
          </motion.div>

          {/* Team Details */}
          {selectedTeam && (
            <motion.div
              variants={fadeUp}
              className="lg:col-span-2 border border-white/[.06] p-6"
            >
              <h2 className="font-display text-lg font-bold text-white mb-4 uppercase">
                {selectedTeam.name}
              </h2>

              <div className="space-y-6">
                <div>
                  <p className="font-mono text-[10px] text-white/30 mb-2 uppercase tracking-[0.1em]">
                    Description
                  </p>
                  <p className="font-mono text-[11px] text-white/60">
                    {selectedTeam.description || 'No description'}
                  </p>
                </div>

                <div>
                  <p className="font-mono text-[10px] text-white/30 mb-3 uppercase tracking-[0.1em]">
                    Members ({members.length})
                  </p>
                  <div className="space-y-2">
                    {members.length === 0 ? (
                      <p className="font-mono text-[11px] text-white/30">No members</p>
                    ) : (
                      members.map((member, idx) => (
                        <div
                          key={idx}
                          className="flex justify-between items-center py-2 px-3 border border-white/[.06]"
                        >
                          <span className="font-mono text-[11px] text-white/60">{member.email}</span>
                          <span className="font-mono text-[10px] text-[#B7FF4A] uppercase">
                            {member.role || 'Member'}
                          </span>
                        </div>
                      ))
                    )}
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </div>
      </motion.div>
    </DashboardLayout>
  );
}
