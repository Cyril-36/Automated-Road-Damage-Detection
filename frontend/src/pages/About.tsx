import { motion } from 'framer-motion'
import AnimatedSection from '../components/AnimatedSection'
import GlassCard from '../components/GlassCard'

const architecture = [
  {
    title: 'RDD2022 Dataset',
    desc: '38,385 road images from 7 countries with YOLO-format annotations across 5 damage categories.',
  },
  {
    title: 'YOLOv8 Models',
    desc: 'Three models at different scales: YOLOv8n@640, YOLOv8s@640, and YOLOv8s@1024.',
  },
  {
    title: 'Weighted Boxes Fusion',
    desc: 'Merges overlapping predictions using configurable weights [1.0, 1.5, 2.0] and IoU threshold 0.5.',
  },
  {
    title: 'Three Inference Modes',
    desc: 'Fast (single model, ~20ms), Balanced (two models, ~100ms), and Accurate (full ensemble).',
  },
  {
    title: 'Global Coverage',
    desc: 'Trained on roads from Japan, India, USA, Norway, China, Czech Republic, and more.',
  },
  {
    title: 'REST API + Web UI',
    desc: 'Flask REST endpoints for integration and this React dashboard for production use.',
  },
]

const damageClasses = [
  { code: 'D00', name: 'Longitudinal Crack', desc: 'Cracks parallel to the road direction' },
  { code: 'D10', name: 'Transverse Crack', desc: 'Cracks perpendicular to the road direction' },
  { code: 'D20', name: 'Alligator Crack', desc: 'Interconnected crack patterns' },
  { code: 'D40', name: 'Pothole', desc: 'Bowl-shaped holes in the road surface' },
  { code: 'D50', name: 'Other Corruption', desc: 'Additional surface deterioration' },
]

const weights = [
  { model: 'YOLOv8n @ 640', weight: '1.0', role: 'Fast baseline' },
  { model: 'YOLOv8s @ 640', weight: '1.5', role: 'Balanced' },
  { model: 'YOLOv8s @ 1024', weight: '2.0', role: 'High resolution' },
]

export default function About() {
  return (
    <div className="mx-auto max-w-3xl px-6 pt-28 pb-20">
      {/* Header */}
      <AnimatedSection>
        <p className="text-[12px] font-medium uppercase tracking-[0.15em] text-accent">About</p>
        <h1 className="mt-2 text-2xl font-semibold tracking-[-0.025em] text-text-primary">
          How it works
        </h1>
        <p className="mt-3 text-[14px] leading-relaxed text-text-secondary max-w-xl">
          A multi-resolution YOLOv8 ensemble with Weighted Boxes Fusion,
          achieving 66.18% mAP@50 on the RDD2022 benchmark.
        </p>
      </AnimatedSection>

      {/* Architecture */}
      <section className="mt-20">
        <AnimatedSection>
          <h2 className="text-[12px] font-medium uppercase tracking-[0.15em] text-text-tertiary">
            Architecture
          </h2>
        </AnimatedSection>

        <div className="mt-5 grid gap-px rounded-xl border border-border bg-border sm:grid-cols-2 overflow-hidden">
          {architecture.map((item, i) => (
            <AnimatedSection key={item.title} delay={i * 0.04}>
              <div className="flex h-full flex-col bg-bg p-5 transition-colors duration-200 hover:bg-bg-elevated">
                <h3 className="text-[13px] font-medium text-text-primary">{item.title}</h3>
                <p className="mt-2 text-[13px] leading-relaxed text-text-secondary">{item.desc}</p>
              </div>
            </AnimatedSection>
          ))}
        </div>
      </section>

      {/* Damage Classes */}
      <section className="mt-20">
        <AnimatedSection>
          <h2 className="text-[12px] font-medium uppercase tracking-[0.15em] text-text-tertiary">
            Damage Classes
          </h2>
        </AnimatedSection>

        <div className="mt-5 rounded-xl border border-border overflow-hidden">
          {damageClasses.map((dc, i) => (
            <AnimatedSection key={dc.code} delay={i * 0.03}>
              <div className={`flex items-center gap-4 px-5 py-4 transition-colors hover:bg-bg-elevated ${
                i < damageClasses.length - 1 ? 'border-b border-border' : ''
              }`}>
                <span className="font-mono text-[12px] font-medium text-accent w-8">{dc.code}</span>
                <div className="flex-1">
                  <p className="text-[13px] font-medium text-text-primary">{dc.name}</p>
                  <p className="text-[12px] text-text-tertiary">{dc.desc}</p>
                </div>
              </div>
            </AnimatedSection>
          ))}
        </div>
      </section>

      {/* WBF */}
      <section className="mt-20">
        <AnimatedSection>
          <h2 className="text-[12px] font-medium uppercase tracking-[0.15em] text-text-tertiary">
            Fusion Strategy
          </h2>
        </AnimatedSection>

        <AnimatedSection className="mt-5">
          <GlassCard hover={false} className="p-6">
            <div className="grid gap-6 sm:grid-cols-3">
              {weights.map((w) => (
                <div key={w.model} className="text-center">
                  <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-full border border-border bg-bg-subtle">
                    <span className="font-mono text-[14px] font-semibold text-accent">{w.weight}</span>
                  </div>
                  <p className="mt-3 text-[13px] font-medium text-text-primary">{w.model}</p>
                  <p className="mt-0.5 text-[11px] text-text-tertiary">{w.role}</p>
                </div>
              ))}
            </div>

            <div className="mt-6 flex flex-wrap items-center justify-center gap-2">
              {['IoU: 0.5', 'Skip: 0.01', 'Conf: avg'].map((tag) => (
                <span key={tag} className="rounded-md border border-border bg-bg-subtle px-2.5 py-1 text-[11px] font-mono text-text-tertiary">
                  {tag}
                </span>
              ))}
              <span className="rounded-md bg-accent-soft px-2.5 py-1 text-[11px] font-mono font-medium text-accent">
                66.18% mAP@50
              </span>
            </div>
          </GlassCard>
        </AnimatedSection>
      </section>

      {/* Tech Stack */}
      <section className="mt-20">
        <AnimatedSection>
          <h2 className="text-[12px] font-medium uppercase tracking-[0.15em] text-text-tertiary">
            Stack
          </h2>
        </AnimatedSection>

        <AnimatedSection className="mt-5">
          <div className="flex flex-wrap gap-2">
            {[
              'PyTorch', 'Ultralytics', 'OpenCV', 'ensemble-boxes',
              'Flask', 'React', 'Tailwind CSS', 'Framer Motion', 'Recharts',
            ].map((tech) => (
              <motion.span
                key={tech}
                whileHover={{ scale: 1.03 }}
                className="rounded-md border border-border px-3 py-1.5 text-[12px] text-text-secondary transition-colors hover:border-border-hover hover:text-text-primary"
              >
                {tech}
              </motion.span>
            ))}
          </div>
        </AnimatedSection>
      </section>
    </div>
  )
}
