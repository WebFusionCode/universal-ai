import React, { useEffect, useRef, useCallback, useState } from 'react';
import { Link } from 'react-router-dom';
import { motion, useScroll, useTransform, useInView } from 'framer-motion';

/* ============================================================
   PARTICLE VORTEX CANVAS — reacts to mouse + scroll
   ============================================================ */
function ParticleVortex() {
  const canvasRef = useRef(null);
  const animRef = useRef(null);
  const scrollY = useRef(0);
  const mouse = useRef({ x: 0.5, y: 0.5, active: false });

  useEffect(() => {
    const handleScroll = () => { scrollY.current = window.scrollY; };
    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let W, H;
    const resize = () => { W = canvas.width = canvas.offsetWidth; H = canvas.height = canvas.offsetHeight; };
    resize();
    window.addEventListener('resize', resize);

    const handleMouseMove = (e) => {
      const rect = canvas.getBoundingClientRect();
      mouse.current.x = (e.clientX - rect.left) / W;
      mouse.current.y = (e.clientY - rect.top) / H;
      mouse.current.active = true;
    };
    const handleMouseLeave = () => { mouse.current.active = false; };
    canvas.addEventListener('mousemove', handleMouseMove);
    canvas.addEventListener('mouseleave', handleMouseLeave);

    const NUM = 700;
    const particles = [];
    for (let i = 0; i < NUM; i++) {
      const angle = Math.random() * Math.PI * 2;
      const radius = 30 + Math.random() * 220;
      particles.push({
        angle, radius, baseRadius: radius,
        speed: 0.001 + Math.random() * 0.005,
        yOff: (Math.random() - 0.5) * 300,
        size: 0.8 + Math.random() * 2.8,
        baseSize: 0.8 + Math.random() * 2.8,
        alpha: 0.12 + Math.random() * 0.55,
        baseAlpha: 0.12 + Math.random() * 0.55,
        dx: 0, dy: 0,
      });
    }

    const draw = () => {
      ctx.clearRect(0, 0, W, H);
      const cx = W / 2;
      const cy = H / 2;
      const scrollOffset = scrollY.current * 0.001;
      const mx = mouse.current.active ? mouse.current.x * W : cx;
      const my = mouse.current.active ? mouse.current.y * H : cy;

      for (const p of particles) {
        p.angle += p.speed;
        const r = p.baseRadius + Math.sin(p.angle * 2 + scrollOffset) * 25;
        let x = cx + Math.cos(p.angle + scrollOffset) * r;
        let y = cy + p.yOff + Math.sin(p.angle * 1.5 + scrollOffset) * 35;

        // Mouse interaction - particles are attracted/repelled
        if (mouse.current.active) {
          const ddx = x - mx;
          const ddy = y - my;
          const dist = Math.sqrt(ddx * ddx + ddy * ddy);
          const maxDist = 200;
          if (dist < maxDist) {
            const force = (1 - dist / maxDist) * 60;
            // Inner particles push away, outer ones attract
            const direction = dist < 80 ? 1 : -0.5;
            p.dx += (ddx / dist) * force * direction * 0.08;
            p.dy += (ddy / dist) * force * direction * 0.08;
            // Glow up near cursor
            p.size = p.baseSize + (1 - dist / maxDist) * 3;
            p.alpha = Math.min(p.baseAlpha + (1 - dist / maxDist) * 0.4, 1);
          } else {
            p.size += (p.baseSize - p.size) * 0.05;
            p.alpha += (p.baseAlpha - p.alpha) * 0.05;
          }
        } else {
          p.size += (p.baseSize - p.size) * 0.03;
          p.alpha += (p.baseAlpha - p.alpha) * 0.03;
        }

        // Apply velocity with damping
        x += p.dx;
        y += p.dy;
        p.dx *= 0.92;
        p.dy *= 0.92;

        // Draw particle with glow
        const glowAlpha = p.alpha * 0.3;
        ctx.beginPath();
        ctx.arc(x, y, p.size + 2, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(183, 255, 74, ${glowAlpha * (1 - scrollY.current * 0.0004)})`;
        ctx.fill();

        ctx.beginPath();
        ctx.arc(x, y, p.size, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(210, 215, 205, ${p.alpha * (1 - scrollY.current * 0.0004)})`;
        ctx.fill();
      }

      // Draw connecting lines near mouse
      if (mouse.current.active) {
        const nearParticles = [];
        for (const p of particles) {
          const r = p.baseRadius + Math.sin(p.angle * 2 + scrollOffset) * 25;
          const x = cx + Math.cos(p.angle + scrollOffset) * r + p.dx;
          const y = cy + p.yOff + Math.sin(p.angle * 1.5 + scrollOffset) * 35 + p.dy;
          const dist = Math.sqrt((x - mx) ** 2 + (y - my) ** 2);
          if (dist < 150) nearParticles.push({ x, y, dist });
        }
        for (let i = 0; i < nearParticles.length; i++) {
          for (let j = i + 1; j < Math.min(nearParticles.length, i + 4); j++) {
            const a = nearParticles[i], b = nearParticles[j];
            const d = Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
            if (d < 80) {
              ctx.beginPath();
              ctx.moveTo(a.x, a.y);
              ctx.lineTo(b.x, b.y);
              ctx.strokeStyle = `rgba(183, 255, 74, ${0.08 * (1 - d / 80)})`;
              ctx.lineWidth = 0.5;
              ctx.stroke();
            }
          }
        }
      }

      animRef.current = requestAnimationFrame(draw);
    };
    draw();
    return () => {
      window.removeEventListener('resize', resize);
      canvas.removeEventListener('mousemove', handleMouseMove);
      canvas.removeEventListener('mouseleave', handleMouseLeave);
      cancelAnimationFrame(animRef.current);
    };
  }, []);

  return <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" style={{ cursor: 'crosshair' }} />;
}

/* ============================================================
   SPARKLE PARTICLES — floating, pulsing, connecting sparkles
   ============================================================ */
function ParticleDots({ color = 'rgba(0,0,0,0.12)', count = 120 }) {
  const canvasRef = useRef(null);
  const animRef = useRef(null);
  const mouse = useRef({ x: -1000, y: -1000 });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let W, H;
    const resize = () => { W = canvas.width = canvas.offsetWidth; H = canvas.height = canvas.offsetHeight; };
    resize();
    window.addEventListener('resize', resize);

    const handleMouseMove = (e) => {
      const rect = canvas.getBoundingClientRect();
      mouse.current.x = e.clientX - rect.left;
      mouse.current.y = e.clientY - rect.top;
    };
    const handleMouseLeave = () => { mouse.current.x = -1000; mouse.current.y = -1000; };
    canvas.addEventListener('mousemove', handleMouseMove);
    canvas.addEventListener('mouseleave', handleMouseLeave);

    const sparkles = [];
    for (let i = 0; i < count; i++) {
      sparkles.push({
        x: Math.random() * W,
        y: Math.random() * H,
        vx: (Math.random() - 0.5) * 0.6,
        vy: (Math.random() - 0.5) * 0.6,
        baseR: 1.2 + Math.random() * 2.5,
        r: 1.2 + Math.random() * 2.5,
        pulsePhase: Math.random() * Math.PI * 2,
        pulseSpeed: 0.02 + Math.random() * 0.03,
        alpha: 0.2 + Math.random() * 0.5,
        trail: [],
      });
    }

    const draw = () => {
      ctx.clearRect(0, 0, W, H);
      const t = Date.now() * 0.001;

      for (const s of sparkles) {
        // Pulsing size
        s.r = s.baseR + Math.sin(t * 3 + s.pulsePhase) * 1;

        // Mouse repulsion
        const dx = s.x - mouse.current.x;
        const dy = s.y - mouse.current.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 120 && dist > 0) {
          const force = (1 - dist / 120) * 2;
          s.vx += (dx / dist) * force * 0.3;
          s.vy += (dy / dist) * force * 0.3;
          s.r = s.baseR + (1 - dist / 120) * 4;
        }

        // Move
        s.x += s.vx;
        s.y += s.vy;
        s.vx *= 0.98;
        s.vy *= 0.98;

        // Wrap around edges
        if (s.x < -10) s.x = W + 10;
        if (s.x > W + 10) s.x = -10;
        if (s.y < -10) s.y = H + 10;
        if (s.y > H + 10) s.y = -10;

        // Trail
        s.trail.push({ x: s.x, y: s.y });
        if (s.trail.length > 5) s.trail.shift();

        // Draw trail
        for (let i = 0; i < s.trail.length - 1; i++) {
          const a = (i / s.trail.length) * s.alpha * 0.3;
          ctx.beginPath();
          ctx.arc(s.trail[i].x, s.trail[i].y, s.r * 0.4, 0, Math.PI * 2);
          ctx.fillStyle = color.replace(/[\d.]+\)$/, `${a})`);
          ctx.fill();
        }

        // Draw sparkle glow
        const glowR = s.r * 3;
        const grad = ctx.createRadialGradient(s.x, s.y, 0, s.x, s.y, glowR);
        grad.addColorStop(0, color.replace(/[\d.]+\)$/, `${s.alpha * 0.4})`));
        grad.addColorStop(1, color.replace(/[\d.]+\)$/, '0)'));
        ctx.beginPath();
        ctx.arc(s.x, s.y, glowR, 0, Math.PI * 2);
        ctx.fillStyle = grad;
        ctx.fill();

        // Draw core
        ctx.beginPath();
        ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
      }

      // Connect nearby sparkles
      for (let i = 0; i < sparkles.length; i++) {
        for (let j = i + 1; j < sparkles.length; j++) {
          const a = sparkles[i], b = sparkles[j];
          const d = Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
          if (d < 100) {
            ctx.beginPath();
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.strokeStyle = color.replace(/[\d.]+\)$/, `${0.06 * (1 - d / 100)})`);
            ctx.lineWidth = 0.5;
            ctx.stroke();
          }
        }
      }

      animRef.current = requestAnimationFrame(draw);
    };
    draw();
    return () => {
      window.removeEventListener('resize', resize);
      canvas.removeEventListener('mousemove', handleMouseMove);
      canvas.removeEventListener('mouseleave', handleMouseLeave);
      cancelAnimationFrame(animRef.current);
    };
  }, [color, count]);

  return <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" />;
}

/* ============================================================
   SCROLL TEXT REVEAL — words animate in on scroll
   ============================================================ */
function RevealText({ text, className = '', as = 'h2', delay = 0 }) {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: '-80px' });
  const words = text.split(' ');
  const Tag = as;
  return (
    <Tag ref={ref} className={className}>
      {words.map((word, i) => (
        <span key={i} className="inline-block overflow-hidden mr-[0.3em]">
          <motion.span
            className="inline-block"
            initial={{ y: '110%', opacity: 0 }}
            animate={inView ? { y: 0, opacity: 1 } : {}}
            transition={{ duration: 0.5, delay: delay + i * 0.06, ease: [0.22, 1, 0.36, 1] }}>
            {word}
          </motion.span>
        </span>
      ))}
    </Tag>
  );
}

/* ============================================================
   FADE-IN ON SCROLL
   ============================================================ */
function FadeIn({ children, className = '', delay = 0, direction = 'up' }) {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: '-60px' });
  const variants = {
    hidden: { opacity: 0, y: direction === 'up' ? 40 : direction === 'down' ? -40 : 0, x: direction === 'left' ? 40 : direction === 'right' ? -40 : 0 },
    visible: { opacity: 1, y: 0, x: 0 },
  };
  return (
    <motion.div ref={ref} className={className}
      initial="hidden" animate={inView ? 'visible' : 'hidden'}
      variants={variants} transition={{ duration: 0.7, delay, ease: [0.22, 1, 0.36, 1] }}>
      {children}
    </motion.div>
  );
}

/* ============================================================
   BRACKET BUTTON COMPONENT
   ============================================================ */
function BracketButton({ children, to, variant = 'light', onClick, className = '' }) {
  const inner = (
    <span className={`bracket-btn ${variant === 'dark' ? 'bracket-btn-dark' : 'bracket-btn-light'} ${className}`}
      onClick={onClick}>
      {children}
    </span>
  );
  if (to) return <Link to={to}>{inner}</Link>;
  return inner;
}

/* ============================================================
   NAVBAR
   ============================================================ */
function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  useEffect(() => {
    const h = () => setScrolled(window.scrollY > 80);
    window.addEventListener('scroll', h, { passive: true });
    return () => window.removeEventListener('scroll', h);
  }, []);

  return (
    <motion.nav initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.8, delay: 0.3 }}
      className={`fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-8 md:px-12 py-5 transition-all duration-500 ${scrolled ? 'bg-[#0a0a0a]/90 backdrop-blur-lg' : ''}`}>
      <Link to="/" className="font-display text-sm tracking-[0.2em] uppercase font-bold text-white">
        AutoML<br /><span className="font-normal tracking-[0.25em]" style={{ color: 'var(--accent-green)' }}>X</span>
      </Link>
      <div className="hidden md:flex items-center gap-10">
        {['Features', 'How It Works', 'Builder', 'Contact'].map((item) => (
          <a key={item} href={`#${item.toLowerCase().replace(/\s/g, '-')}`}
            className="font-mono text-[11px] tracking-[0.15em] uppercase text-white/50 hover:text-white transition-colors duration-300">
            {item}
          </a>
        ))}
      </div>
      <div className="flex items-center gap-4">
        <Link to="/login" className="font-mono text-[11px] tracking-[0.15em] uppercase text-white/50 hover:text-white transition-colors" data-testid="nav-login">Login</Link>
        <Link to="/signup" data-testid="nav-get-started">
          <span className="bracket-btn bracket-btn-light text-[10px] py-2 px-5">Get Started</span>
        </Link>
      </div>
    </motion.nav>
  );
}

/* ============================================================
   HERO — Split text + particle vortex center
   ============================================================ */
function Hero() {
  const ref = useRef(null);
  const { scrollYProgress } = useScroll({ target: ref, offset: ['start start', 'end start'] });
  const opacity = useTransform(scrollYProgress, [0, 0.6], [1, 0]);
  const y1 = useTransform(scrollYProgress, [0, 0.6], [0, -100]);
  const y2 = useTransform(scrollYProgress, [0, 0.6], [0, -60]);

  return (
    <section ref={ref} className="relative min-h-screen flex flex-col justify-center section-dark overflow-hidden">
      {/* Particle Vortex - full screen, interactive */}
      <div className="absolute inset-0">
        <ParticleVortex />
      </div>

      <motion.div style={{ opacity }} className="relative z-10 px-8 md:px-16 lg:px-24 pointer-events-none">
        <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-8 max-w-[1400px] mx-auto">
          {/* Left text */}
          <motion.div style={{ y: y1 }}>
            <h1 className="font-display text-[clamp(2.5rem,6vw,5.5rem)] font-bold leading-[0.95] tracking-tight uppercase">
              Build AI<br />Models That
            </h1>
          </motion.div>
          {/* Right text */}
          <motion.div style={{ y: y2 }} className="lg:text-right">
            <h1 className="font-display text-[clamp(2.5rem,6vw,5.5rem)] font-bold leading-[0.95] tracking-tight uppercase">
              <span style={{ color: 'var(--accent-green)' }}>Define</span><br />The Future
            </h1>
          </motion.div>
        </div>
      </motion.div>

      {/* Bottom section */}
      <motion.div style={{ opacity }} className="absolute bottom-12 left-0 right-0 z-10 px-8 md:px-16 lg:px-24 pointer-events-none">
        <div className="flex flex-col md:flex-row items-end justify-between gap-8 max-w-[1400px] mx-auto">
          <div className="pointer-events-auto">
            <BracketButton to="/signup" variant="light">
              Get Started
            </BracketButton>
          </div>
          <p className="font-mono text-[11px] leading-relaxed tracking-wide text-white/40 max-w-md uppercase">
            AutoML X is an intelligent platform that trains, evaluates, and deploys machine learning models automatically. Upload data. Get results. No code required.
          </p>
        </div>
      </motion.div>
    </section>
  );
}

/* ============================================================
   ETHOS SECTION — "Vision Matters. Velocity Wins." (Light bg)
   ============================================================ */
function Ethos() {
  return (
    <section className="section-light py-28 px-8 md:px-16 lg:px-24">
      <div className="max-w-[1400px] mx-auto">
        <div className="flex flex-col lg:flex-row gap-16 mb-20">
          <div className="lg:w-1/2">
            <FadeIn>
              <p className="font-mono text-[11px] tracking-[0.2em] uppercase text-black/40 mb-4">Our Platform</p>
            </FadeIn>
            <RevealText
              text="Intelligence Matters. Speed Wins."
              className="font-display text-[clamp(2rem,4vw,3.5rem)] font-bold leading-[1.05] tracking-tight uppercase text-[#0a0a0a]"
            />
          </div>
          <div className="lg:w-1/2 lg:pt-8">
            <FadeIn delay={0.2}>
              <div className="w-full h-[1px] bg-black/10 mb-6" />
              <p className="font-body text-[15px] leading-relaxed text-black/60 mb-8">
                Our comprehensive AutoML platform shifts the odds. With infrastructure that works. With algorithms that compete. With the right pressure — pushing your models forward, not under. This isn't a notebook. It's complete model building, at the speed AI demands.
              </p>
              <BracketButton to="/signup" variant="dark">Join The Platform</BracketButton>
            </FadeIn>
          </div>
        </div>

        {/* 4 Feature Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4" id="features">
          {[
            { title: 'AutoML\nTraining', bg: '#0a0a0a', color: '#fff', num: '01', desc: 'Multiple algorithms compete. The best model wins. Automated hyperparameter tuning included.' },
            { title: 'Dataset\nAnalysis', bg: 'var(--accent-teal)', color: '#fff', num: '02', desc: 'Upload CSV, Excel, or JSON. Instant quality scores, type detection, and smart previews.' },
            { title: 'AI\nAssistant', bg: 'var(--accent-orange)', color: '#fff', num: '03', desc: 'LLM-powered advisor for model selection, feature engineering, and debugging guidance.' },
            { title: 'Model\nExport', bg: 'var(--bg-light)', color: '#0a0a0a', num: '04', desc: 'Download trained models, Python pipelines, and Jupyter notebooks. Production ready.' },
          ].map((card, i) => (
            <FadeIn key={i} delay={i * 0.1}>
              <div className="relative overflow-hidden group cursor-default h-[380px] flex flex-col justify-between p-7 transition-transform duration-500 hover:-translate-y-1"
                style={{ background: card.bg, color: card.color }}>
                {/* Geometric dot art */}
                <div className="absolute bottom-0 right-0 w-40 h-40 opacity-20">
                  <svg viewBox="0 0 160 160" fill="none">
                    {Array.from({ length: 8 }).map((_, row) =>
                      Array.from({ length: 8 }).map((_, col) => (
                        <circle key={`${row}-${col}`} cx={20 + col * 18} cy={20 + row * 18} r={2 + Math.sin(row + col) * 1.5}
                          fill="currentColor" opacity={0.3 + Math.random() * 0.4} />
                      ))
                    )}
                  </svg>
                </div>
                <div>
                  <h3 className="font-display text-2xl font-bold uppercase leading-tight tracking-tight whitespace-pre-line">{card.title}</h3>
                </div>
                <div>
                  <p className="font-mono text-[10px] tracking-[0.15em] uppercase opacity-50 mb-3">{card.num} / 04</p>
                  <p className="text-[13px] leading-relaxed opacity-70">{card.desc}</p>
                </div>
              </div>
            </FadeIn>
          ))}
        </div>
      </div>
    </section>
  );
}

/* ============================================================
   FOCUS SECTION — Particle dot background (Light bg)
   ============================================================ */
function Focus() {
  return (
    <section className="section-light relative py-36 px-8 md:px-16 overflow-hidden" id="how-it-works">
      <ParticleDots color="rgba(0,0,0,0.10)" count={150} />
      <div className="relative z-10 max-w-[1000px] mx-auto text-center">
        <FadeIn>
          <p className="font-mono text-[11px] tracking-[0.2em] uppercase text-black/40 mb-6">How It Works</p>
        </FadeIn>
        <RevealText
          text="Upload. Train. Deploy. Four steps to production-ready AI."
          className="font-display text-[clamp(1.8rem,4vw,3.2rem)] font-bold leading-[1.1] tracking-tight uppercase text-[#0a0a0a]"
        />
        <FadeIn delay={0.3}>
          <p className="font-body text-[15px] leading-relaxed text-black/50 mt-8 max-w-xl mx-auto">
            CSV, Excel, JSON — these aren't just file formats. They're the starting point for models that solve real problems. Upload. Click train. We handle everything else.
          </p>
        </FadeIn>
      </div>

      {/* Steps */}
      <div className="relative z-10 max-w-[1200px] mx-auto mt-24 grid grid-cols-1 md:grid-cols-4 gap-8">
        {[
          { num: '01', title: 'Upload Dataset', desc: 'Drag & drop your data file. We parse it instantly.' },
          { num: '02', title: 'Auto Analysis', desc: 'Quality scores, type detection, feature recommendations.' },
          { num: '03', title: 'Train Models', desc: 'Multiple algorithms compete. Best model auto-selected.' },
          { num: '04', title: 'Deploy & Predict', desc: 'Download models or run predictions immediately.' },
        ].map((step, i) => (
          <FadeIn key={i} delay={i * 0.12}>
            <div className="text-center">
              <p className="font-mono text-[48px] font-bold text-black/[.06] mb-2">{step.num}</p>
              <h3 className="font-display text-lg font-bold uppercase tracking-tight text-[#0a0a0a] mb-2">{step.title}</h3>
              <p className="text-[13px] text-black/50 leading-relaxed">{step.desc}</p>
            </div>
          </FadeIn>
        ))}
      </div>
    </section>
  );
}

/* ============================================================
   BUILDER PREVIEW — Dark section with dashboard mockup
   ============================================================ */
function BuilderPreview() {
  return (
    <section className="section-dark py-32 px-8 md:px-16 overflow-hidden" id="builder">
      <div className="max-w-[1400px] mx-auto">
        <div className="flex flex-col lg:flex-row items-start justify-between gap-12 mb-16">
          <div className="lg:w-1/2">
            <FadeIn>
              <p className="font-mono text-[11px] tracking-[0.2em] uppercase text-white/40 mb-4">The Dashboard</p>
            </FadeIn>
            <RevealText
              text="Your complete AI workspace. Train. Experiment. Export."
              className="font-display text-[clamp(1.8rem,3.5vw,2.8rem)] font-bold leading-[1.1] tracking-tight uppercase text-white"
            />
          </div>
          <FadeIn delay={0.2} className="lg:w-1/3 lg:pt-6">
            <p className="font-body text-[14px] leading-relaxed text-white/40">
              A fixed-layout workspace designed for data scientists. Upload datasets, monitor training in real-time, compare models on leaderboards, and export everything.
            </p>
          </FadeIn>
        </div>

        {/* Dashboard Mockup */}
        <FadeIn delay={0.3}>
          <div className="relative border border-white/[.08] bg-[#111] overflow-hidden">
            {/* Top bar */}
            <div className="flex items-center justify-between px-5 py-3 border-b border-white/[.06]">
              <div className="flex items-center gap-2">
                <div className="w-2.5 h-2.5 rounded-full bg-[#FF5C7A]" />
                <div className="w-2.5 h-2.5 rounded-full bg-[#FFCC66]" />
                <div className="w-2.5 h-2.5 rounded-full bg-[#B7FF4A]" />
              </div>
              <p className="font-mono text-[10px] tracking-wider text-white/25 uppercase">automl-x.app/train</p>
              <div className="w-12" />
            </div>
            <div className="flex min-h-[360px]">
              {/* Sidebar */}
              <div className="hidden md:flex flex-col w-44 border-r border-white/[.06] py-4 px-3 gap-0.5">
                {['Dashboard', 'Train Model', 'Experiments', 'Leaderboard', 'Predict', 'Downloads'].map((item, i) => (
                  <div key={i} className={`px-3 py-2 font-mono text-[10px] tracking-wider uppercase ${i === 1 ? 'bg-white/[.06] text-[#B7FF4A]' : 'text-white/30 hover:text-white/50'} transition-colors`}>
                    {item}
                  </div>
                ))}
              </div>
              {/* Main */}
              <div className="flex-1 p-6 space-y-5">
                <div className="flex items-center gap-3 flex-wrap">
                  <div className="bg-white/[.04] border border-white/[.06] px-4 py-2 font-mono text-[10px] text-white/50 uppercase tracking-wider">loan_data.csv</div>
                  <div className="bg-white/[.04] border border-white/[.06] px-4 py-2 font-mono text-[10px] text-[#B7FF4A] uppercase tracking-wider">12,500 rows</div>
                  <div className="bg-white/[.04] border border-white/[.06] px-4 py-2 font-mono text-[10px] text-[#6AA7FF] uppercase tracking-wider">42 features</div>
                </div>
                {/* Table */}
                <div className="border border-white/[.06] overflow-hidden">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b border-white/[.06]">
                        {['ID', 'Income', 'Credit_Score', 'Approved'].map((h, i) => (
                          <th key={i} className="px-4 py-2.5 text-left font-mono text-[10px] text-white/30 uppercase tracking-wider font-normal">{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {[
                        [1, '85,400', '742', 'Yes'],
                        [2, '42,100', '615', 'No'],
                        [3, '126,800', '801', 'Yes'],
                      ].map((row, r) => (
                        <tr key={r} className="border-b border-white/[.04]">
                          {row.map((cell, c) => (
                            <td key={c} className={`px-4 py-2 font-mono text-[11px] ${c === 3 ? (cell === 'Yes' ? 'text-[#B7FF4A]' : 'text-[#FF5C7A]') : 'text-white/50'}`}>{cell}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                {/* Progress */}
                <div className="border border-white/[.06] p-4">
                  <div className="flex justify-between items-center mb-3">
                    <span className="font-mono text-[10px] text-white/40 uppercase tracking-wider">Training Progress</span>
                    <span className="font-mono text-[11px] text-[#B7FF4A] font-bold">87%</span>
                  </div>
                  <div className="h-1.5 bg-white/[.06] overflow-hidden">
                    <motion.div initial={{ width: 0 }} whileInView={{ width: '87%' }} viewport={{ once: true }}
                      transition={{ duration: 2, delay: 0.5 }}
                      className="h-full bg-gradient-to-r from-[#B7FF4A] to-[#6AA7FF]" />
                  </div>
                </div>
              </div>
              {/* Right panel */}
              <div className="hidden lg:flex flex-col w-48 border-l border-white/[.06] py-4 px-3 gap-3">
                <p className="font-mono text-[10px] text-white/30 uppercase tracking-wider mb-1">Model Settings</p>
                {[
                  { label: 'Algorithm', value: 'RandomForest' },
                  { label: 'Estimators', value: '200' },
                  { label: 'Max Depth', value: '12' },
                  { label: 'Problem', value: 'Classification' },
                ].map((s, i) => (
                  <div key={i}>
                    <p className="font-mono text-[9px] text-white/25 uppercase tracking-wider mb-1">{s.label}</p>
                    <div className="bg-white/[.04] border border-white/[.06] px-3 py-1.5 font-mono text-[10px] text-white/60">{s.value}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </FadeIn>
      </div>
    </section>
  );
}

/* ============================================================
   CAPABILITIES SECTION — Horizontal scroll cards
   ============================================================ */
function Capabilities() {
  const scrollRef = useRef(null);
  const capabilities = [
    { title: 'Classification', desc: 'Binary and multi-class prediction with automated encoding.', icon: '01' },
    { title: 'Regression', desc: 'Continuous value prediction with R² optimization.', icon: '02' },
    { title: 'Feature Engineering', desc: 'Auto-generated date, log, interaction, and frequency features.', icon: '03' },
    { title: 'Hyperparameter Tuning', desc: 'Optuna-powered optimization for every model.', icon: '04' },
    { title: 'Model Explainability', desc: 'Feature importance and coefficient analysis built in.', icon: '05' },
    { title: 'Code Generation', desc: 'Auto-generated Python pipelines and Jupyter notebooks.', icon: '06' },
  ];
  return (
    <section className="section-light py-28 px-8 md:px-16">
      <div className="max-w-[1400px] mx-auto">
        <div className="flex items-end justify-between mb-12">
          <div>
            <FadeIn><p className="font-mono text-[11px] tracking-[0.2em] uppercase text-black/40 mb-4">Capabilities</p></FadeIn>
            <RevealText text="What AutoML X can do for you." className="font-display text-[clamp(1.5rem,3vw,2.5rem)] font-bold tracking-tight uppercase text-[#0a0a0a]" />
          </div>
          <FadeIn delay={0.2}>
            <BracketButton to="/signup" variant="dark">Start Building</BracketButton>
          </FadeIn>
        </div>
        <div ref={scrollRef} className="flex gap-4 overflow-x-auto pb-6 snap-x snap-mandatory scrollbar-hide" style={{ scrollbarWidth: 'none' }}>
          {capabilities.map((cap, i) => (
            <FadeIn key={i} delay={i * 0.08}>
              <div className="min-w-[300px] md:min-w-[320px] snap-start border border-black/[.08] p-7 flex flex-col justify-between h-[280px] group hover:bg-black/[.03] transition-all duration-300 cursor-default">
                <div>
                  <p className="font-mono text-[40px] font-bold text-black/[.06] mb-3">{cap.icon}</p>
                  <h3 className="font-display text-lg font-bold uppercase tracking-tight text-[#0a0a0a]">{cap.title}</h3>
                </div>
                <p className="text-[13px] text-black/50 leading-relaxed">{cap.desc}</p>
              </div>
            </FadeIn>
          ))}
        </div>
      </div>
    </section>
  );
}

/* ============================================================
   MARQUEE — Scrolling text
   ============================================================ */
function Marquee() {
  const text = 'AUTOML \u00B7 TRAIN \u00B7 PREDICT \u00B7 EXPORT \u00B7 INTELLIGENCE \u00B7 SCALE \u00B7 DEPLOY \u00B7 ITERATE \u00B7 ';
  return (
    <div className="section-dark py-10 overflow-hidden border-y border-white/[.06]">
      <div className="flex whitespace-nowrap animate-marquee">
        <span className="font-display text-[clamp(3rem,6vw,5rem)] font-bold text-white/[.04] tracking-wider uppercase">
          {text}{text}{text}
        </span>
      </div>
    </div>
  );
}

/* ============================================================
   INVESTORS / CTA — Dark section with particle accent
   ============================================================ */
function CTASection() {
  return (
    <section className="section-dark relative py-36 px-8 md:px-16 overflow-hidden" id="contact">
      <div className="absolute inset-0 opacity-30">
        <ParticleDots color="rgba(183,255,74,0.08)" count={80} />
      </div>
      <div className="relative z-10 max-w-[1200px] mx-auto">
        <div className="flex flex-col lg:flex-row gap-16 items-start">
          <div className="lg:w-1/2">
            <FadeIn>
              <p className="font-mono text-[11px] tracking-[0.2em] uppercase text-white/40 mb-6">For Everyone</p>
            </FadeIn>
            <RevealText
              text="Build production-ready AI models in minutes, not months."
              className="font-display text-[clamp(1.8rem,3.5vw,3rem)] font-bold leading-[1.1] tracking-tight uppercase text-white"
            />
            <FadeIn delay={0.3}>
              <div className="mt-10 flex flex-wrap gap-4">
                <Link to="/signup" className="btn-primary" data-testid="cta-get-started">Get Started Free</Link>
                <BracketButton to="/login" variant="light">Login</BracketButton>
              </div>
            </FadeIn>
          </div>
          <div className="lg:w-1/2 grid grid-cols-1 gap-4">
            {[
              { num: '01', title: 'Zero Code Required', desc: 'Upload your dataset and let AutoML X handle everything from preprocessing to model selection.' },
              { num: '02', title: 'Production Ready', desc: 'Export trained models, generated code, and notebooks. Deploy immediately.' },
              { num: '03', title: 'AI-Powered Guidance', desc: 'Built-in LLM assistant provides expert recommendations for every step of your ML workflow.' },
            ].map((item, i) => (
              <FadeIn key={i} delay={0.15 + i * 0.1}>
                <div className="border border-white/[.08] p-6 hover:border-white/[.15] transition-all duration-300 group">
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="font-display text-base font-bold uppercase tracking-tight text-white">{item.title}</h3>
                    <span className="font-mono text-[10px] text-white/20 tracking-wider">{item.num}</span>
                  </div>
                  <p className="text-[13px] leading-relaxed text-white/40">{item.desc}</p>
                </div>
              </FadeIn>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

/* ============================================================
   FOOTER
   ============================================================ */
function Footer() {
  return (
    <footer className="section-dark border-t border-white/[.06] py-12 px-8 md:px-16">
      <div className="max-w-[1400px] mx-auto flex flex-col md:flex-row justify-between items-center gap-6">
        <div className="font-display text-sm tracking-[0.2em] uppercase font-bold">
          AutoML <span style={{ color: 'var(--accent-green)' }}>X</span>
        </div>
        <div className="flex gap-8">
          {['Features', 'How It Works', 'Builder', 'Login'].map(item => (
            <span key={item} className="font-mono text-[10px] tracking-[0.15em] uppercase text-white/30 hover:text-white/60 cursor-pointer transition-colors">{item}</span>
          ))}
        </div>
        <p className="font-mono text-[10px] tracking-wider text-white/20 uppercase">&copy; 2025 AutoML X</p>
      </div>
    </footer>
  );
}

/* ============================================================
   LANDING PAGE
   ============================================================ */
export default function Landing() {
  return (
    <div className="min-h-screen">
      <Navbar />
      <Hero />
      <Ethos />
      <Focus />
      <BuilderPreview />
      <Capabilities />
      <Marquee />
      <CTASection />
      <Footer />
    </div>
  );
}
