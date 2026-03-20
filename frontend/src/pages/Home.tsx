import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { ArrowRight } from 'lucide-react'
import AnimatedSection from '../components/AnimatedSection'
import GlassCard from '../components/GlassCard'
import GradientButton from '../components/GradientButton'
import CountUp from '../components/CountUp'
import AnimatedGrid from '../components/AnimatedGrid'

const stats = [
  { value: 66.18, suffix: '%', label: 'mAP@50', decimals: 2 },
  { value: 38385, suffix: '+', label: 'Images', decimals: 0 },
  { value: 5, suffix: '', label: 'Classes', decimals: 0 },
  { value: 85, suffix: 'ms', label: 'Latency', decimals: 0 },
]

const features = [
  {
    title: 'Multi-Model Ensemble',
    desc: 'Three YOLOv8 models at different resolutions, fused with Weighted Boxes Fusion.',
  },
  {
    title: 'Real-Time Inference',
    desc: 'From 15ms single-model to 85ms ensemble. Choose speed or accuracy.',
  },
  {
    title: '5 Damage Classes',
    desc: 'Longitudinal cracks, transverse cracks, alligator cracks, potholes, and more.',
  },
  {
    title: 'Global Dataset',
    desc: '38,000+ road images from 7 countries. Robust generalization.',
  },
  {
    title: 'Weighted Boxes Fusion',
    desc: 'Configurable weights and IoU thresholds for optimal prediction merging.',
  },
  {
    title: '3 Detection Modes',
    desc: 'Fast, Balanced, and Accurate — trade latency for precision per request.',
  },
]

const steps = [
  { num: '1', title: 'Upload', desc: 'Drop a road image' },
  { num: '2', title: 'Infer', desc: '3 YOLOv8 models run' },
  { num: '3', title: 'Fuse', desc: 'WBF merges boxes' },
  { num: '4', title: 'Results', desc: 'Annotated output' },
]

export default function Home() {
  return (
    <>
      {/* Hero */}
      <section className="relative overflow-hidden">
        <AnimatedGrid />

        {/* Radial glow */}
        <div className="pointer-events-none absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[500px] bg-accent/[0.06] rounded-full blur-[150px]" />

        <div className="relative mx-auto flex max-w-3xl flex-col items-center px-6 pt-40 pb-32 text-center">
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, ease: [0.25, 0.1, 0, 1] }}
            className="mb-5 inline-flex items-center gap-2 rounded-full border border-border bg-bg-elevated px-3 py-1 text-[12px] text-text-tertiary"
          >
            <span className="h-1.5 w-1.5 rounded-full bg-accent animate-pulse" />
            YOLOv8 Ensemble + Weighted Boxes Fusion
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.08, ease: [0.25, 0.1, 0, 1] }}
            className="text-[clamp(2.2rem,5.5vw,3.8rem)] font-semibold leading-[1.08] tracking-[-0.035em] text-text-primary"
          >
            Detect road damage
            <br />
            <span className="text-text-tertiary">automatically.</span>
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.16, ease: [0.25, 0.1, 0, 1] }}
            className="mt-5 max-w-lg text-[15px] leading-relaxed text-text-secondary"
          >
            Upload road images and instantly identify cracks, potholes, and surface
            damage with a multi-resolution deep learning ensemble.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.26, ease: [0.25, 0.1, 0, 1] }}
            className="mt-8 flex items-center gap-3"
          >
            <Link to="/detect">
              <GradientButton>
                Start Detecting <ArrowRight className="h-3.5 w-3.5" />
              </GradientButton>
            </Link>
            <Link to="/dashboard">
              <GradientButton variant="secondary">View Metrics</GradientButton>
            </Link>
          </motion.div>
        </div>
      </section>

      {/* Stats */}
      <section className="border-y border-border">
        <div className="mx-auto grid max-w-3xl grid-cols-4 divide-x divide-border">
          {stats.map((s, i) => (
            <AnimatedSection key={s.label} delay={i * 0.04} className="px-6 py-10 text-center">
              <p className="text-2xl font-semibold tracking-[-0.02em] text-text-primary">
                <CountUp end={s.value} suffix={s.suffix} decimals={s.decimals} />
              </p>
              <p className="mt-1.5 text-[12px] text-text-tertiary">{s.label}</p>
            </AnimatedSection>
          ))}
        </div>
      </section>

      {/* Features */}
      <section className="mx-auto max-w-3xl px-6 py-28">
        <AnimatedSection className="mb-12">
          <p className="text-[12px] font-medium uppercase tracking-[0.15em] text-accent">Capabilities</p>
          <h2 className="mt-2 text-2xl font-semibold tracking-[-0.025em] text-text-primary">
            Built for precision
          </h2>
        </AnimatedSection>

        <div className="grid gap-px rounded-xl border border-border bg-border sm:grid-cols-2 lg:grid-cols-3 overflow-hidden">
          {features.map((f, i) => (
            <AnimatedSection key={f.title} delay={i * 0.04}>
              <div className="flex h-full flex-col bg-bg p-6 transition-colors duration-200 hover:bg-bg-elevated">
                <h3 className="text-[14px] font-medium text-text-primary">{f.title}</h3>
                <p className="mt-2 text-[13px] leading-relaxed text-text-secondary">{f.desc}</p>
              </div>
            </AnimatedSection>
          ))}
        </div>
      </section>

      {/* Pipeline */}
      <section className="border-t border-border">
        <div className="mx-auto max-w-3xl px-6 py-28">
          <AnimatedSection className="mb-14">
            <p className="text-[12px] font-medium uppercase tracking-[0.15em] text-accent">Pipeline</p>
            <h2 className="mt-2 text-2xl font-semibold tracking-[-0.025em] text-text-primary">
              Four steps to detection
            </h2>
          </AnimatedSection>

          <div className="grid grid-cols-4 gap-6">
            {steps.map((step, i) => (
              <AnimatedSection key={step.num} delay={i * 0.08}>
                <div className="text-center">
                  <div className="mx-auto flex h-10 w-10 items-center justify-center rounded-full border border-border bg-bg-elevated text-[13px] font-semibold text-accent">
                    {step.num}
                  </div>
                  <p className="mt-4 text-[13px] font-medium text-text-primary">{step.title}</p>
                  <p className="mt-1 text-[12px] text-text-tertiary">{step.desc}</p>
                </div>
                {i < 3 && (
                  <div className="absolute right-0 top-5 hidden h-px w-full -translate-x-1/2 bg-border lg:block" />
                )}
              </AnimatedSection>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="border-t border-border">
        <AnimatedSection className="mx-auto max-w-3xl px-6 py-28 text-center">
          <h2 className="text-2xl font-semibold tracking-[-0.025em] text-text-primary">
            Ready to try it?
          </h2>
          <p className="mt-3 text-[14px] text-text-secondary">
            Upload your first image and see the ensemble in action.
          </p>
          <div className="mt-7">
            <Link to="/detect">
              <GradientButton>
                Get Started <ArrowRight className="h-3.5 w-3.5" />
              </GradientButton>
            </Link>
          </div>
        </AnimatedSection>
      </section>
    </>
  )
}
