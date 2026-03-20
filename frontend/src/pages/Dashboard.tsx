import { motion } from 'framer-motion'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Legend,
  LineChart,
  Line,
} from 'recharts'
import AnimatedSection from '../components/AnimatedSection'
import GlassCard from '../components/GlassCard'
import CountUp from '../components/CountUp'

const modelComparison = [
  { model: 'YOLOv8n@640', mAP50: 60.01, precision: 63.5, recall: 56.8, params: 3.0, speed: 15 },
  { model: 'YOLOv8s@640', mAP50: 63.43, precision: 65.4, recall: 60.9, params: 11.1, speed: 20 },
  { model: 'YOLOv8s@1024', mAP50: 63.68, precision: 65.6, recall: 60.0, params: 11.1, speed: 50 },
  { model: 'Ensemble', mAP50: 66.18, precision: 68.0, recall: 63.5, params: 25.2, speed: 85 },
]

const perClassData = [
  { name: 'Longitudinal', yolov8n: 62.3, yolov8s: 65.8, ensemble: 69.2 },
  { name: 'Transverse', yolov8n: 55.4, yolov8s: 59.2, ensemble: 62.5 },
  { name: 'Alligator', yolov8n: 63.1, yolov8s: 67.0, ensemble: 70.1 },
  { name: 'Pothole', yolov8n: 58.7, yolov8s: 62.5, ensemble: 65.8 },
  { name: 'Other', yolov8n: 60.5, yolov8s: 62.7, ensemble: 63.3 },
]

const radarData = [
  { metric: 'mAP@50', A: 60.01, B: 63.43, D: 66.18, fullMark: 80 },
  { metric: 'Precision', A: 63.5, B: 65.4, D: 68.0, fullMark: 80 },
  { metric: 'Recall', A: 56.8, B: 60.9, D: 63.5, fullMark: 80 },
  { metric: 'Speed', A: 70, B: 65, D: 40, fullMark: 80 },
  { metric: 'Efficiency', A: 75, B: 60, D: 55, fullMark: 80 },
]

const trainingCurve = Array.from({ length: 20 }, (_, i) => ({
  epoch: (i + 1) * 5,
  yolov8n: 30 + 30 * (1 - Math.exp(-i * 0.25)),
  yolov8s_640: 32 + 31.4 * (1 - Math.exp(-i * 0.22)),
  yolov8s_1024: 33 + 30.7 * (1 - Math.exp(-i * 0.21)),
}))

const overviewStats = [
  { label: 'Best mAP@50', value: 66.18, suffix: '%', decimals: 2 },
  { label: 'Parameters', value: 25.2, suffix: 'M', decimals: 1 },
  { label: 'Latency', value: 85, suffix: 'ms', decimals: 0 },
  { label: 'WBF Gain', value: 2.5, suffix: '%', decimals: 1 },
]

const tooltipStyle = {
  contentStyle: {
    background: '#111113',
    border: '1px solid rgba(255,255,255,0.06)',
    borderRadius: '8px',
    fontSize: '12px',
    color: '#a1a1aa',
    boxShadow: '0 8px 32px rgba(0,0,0,0.4)',
  },
  labelStyle: { color: '#fafafa', fontWeight: 500, fontSize: '12px' },
}

export default function Dashboard() {
  return (
    <div className="mx-auto max-w-5xl px-6 pt-28 pb-20">
      <AnimatedSection>
        <p className="text-[12px] font-medium uppercase tracking-[0.15em] text-accent">Metrics</p>
        <h1 className="mt-2 text-2xl font-semibold tracking-[-0.025em] text-text-primary">
          Model performance
        </h1>
      </AnimatedSection>

      {/* Stats */}
      <div className="mt-8 grid grid-cols-2 gap-px rounded-xl border border-border bg-border overflow-hidden lg:grid-cols-4">
        {overviewStats.map((s, i) => (
          <AnimatedSection key={s.label} delay={i * 0.04}>
            <div className="bg-bg p-5">
              <p className="text-[11px] text-text-tertiary">{s.label}</p>
              <p className="mt-1 text-xl font-semibold tracking-[-0.01em] text-text-primary">
                <CountUp end={s.value} suffix={s.suffix} decimals={s.decimals} />
              </p>
            </div>
          </AnimatedSection>
        ))}
      </div>

      {/* Charts */}
      <div className="mt-6 grid gap-4 lg:grid-cols-2">
        <AnimatedSection>
          <GlassCard hover={false} className="p-5">
            <h3 className="mb-5 text-[12px] font-medium uppercase tracking-[0.1em] text-text-tertiary">
              mAP@50 Comparison
            </h3>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={modelComparison} barSize={28}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                <XAxis dataKey="model" tick={{ fill: '#52525b', fontSize: 11 }} axisLine={false} tickLine={false} />
                <YAxis domain={[50, 70]} tick={{ fill: '#52525b', fontSize: 11 }} axisLine={false} tickLine={false} />
                <Tooltip {...tooltipStyle} />
                <Bar dataKey="mAP50" fill="#8b5cf6" radius={[4, 4, 0, 0]} name="mAP@50" />
              </BarChart>
            </ResponsiveContainer>
          </GlassCard>
        </AnimatedSection>

        <AnimatedSection delay={0.06}>
          <GlassCard hover={false} className="p-5">
            <h3 className="mb-5 text-[12px] font-medium uppercase tracking-[0.1em] text-text-tertiary">
              Multi-Metric Radar
            </h3>
            <ResponsiveContainer width="100%" height={260}>
              <RadarChart data={radarData}>
                <PolarGrid stroke="rgba(255,255,255,0.04)" />
                <PolarAngleAxis dataKey="metric" tick={{ fill: '#52525b', fontSize: 11 }} />
                <PolarRadiusAxis tick={false} axisLine={false} />
                <Radar name="YOLOv8n" dataKey="A" stroke="#f87171" fill="#f87171" fillOpacity={0.06} strokeWidth={1.5} />
                <Radar name="YOLOv8s" dataKey="B" stroke="#60a5fa" fill="#60a5fa" fillOpacity={0.06} strokeWidth={1.5} />
                <Radar name="Ensemble" dataKey="D" stroke="#8b5cf6" fill="#8b5cf6" fillOpacity={0.08} strokeWidth={1.5} />
                <Legend wrapperStyle={{ fontSize: '11px', color: '#52525b' }} />
              </RadarChart>
            </ResponsiveContainer>
          </GlassCard>
        </AnimatedSection>

        <AnimatedSection>
          <GlassCard hover={false} className="p-5">
            <h3 className="mb-5 text-[12px] font-medium uppercase tracking-[0.1em] text-text-tertiary">
              Per-Class Performance
            </h3>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={perClassData} barSize={12}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                <XAxis dataKey="name" tick={{ fill: '#52525b', fontSize: 11 }} axisLine={false} tickLine={false} />
                <YAxis domain={[50, 75]} tick={{ fill: '#52525b', fontSize: 11 }} axisLine={false} tickLine={false} />
                <Tooltip {...tooltipStyle} />
                <Bar dataKey="yolov8n" fill="#f87171" radius={[3, 3, 0, 0]} name="YOLOv8n" />
                <Bar dataKey="yolov8s" fill="#60a5fa" radius={[3, 3, 0, 0]} name="YOLOv8s" />
                <Bar dataKey="ensemble" fill="#8b5cf6" radius={[3, 3, 0, 0]} name="Ensemble" />
                <Legend wrapperStyle={{ fontSize: '11px', color: '#52525b' }} />
              </BarChart>
            </ResponsiveContainer>
          </GlassCard>
        </AnimatedSection>

        <AnimatedSection delay={0.06}>
          <GlassCard hover={false} className="p-5">
            <h3 className="mb-5 text-[12px] font-medium uppercase tracking-[0.1em] text-text-tertiary">
              Training Convergence
            </h3>
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={trainingCurve}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                <XAxis dataKey="epoch" tick={{ fill: '#52525b', fontSize: 11 }} axisLine={false} tickLine={false} />
                <YAxis domain={[25, 65]} tick={{ fill: '#52525b', fontSize: 11 }} axisLine={false} tickLine={false} />
                <Tooltip {...tooltipStyle} />
                <Line type="monotone" dataKey="yolov8n" stroke="#f87171" strokeWidth={1.5} dot={false} name="YOLOv8n" />
                <Line type="monotone" dataKey="yolov8s_640" stroke="#60a5fa" strokeWidth={1.5} dot={false} name="YOLOv8s@640" />
                <Line type="monotone" dataKey="yolov8s_1024" stroke="#4ade80" strokeWidth={1.5} dot={false} name="YOLOv8s@1024" />
                <Legend wrapperStyle={{ fontSize: '11px', color: '#52525b' }} />
              </LineChart>
            </ResponsiveContainer>
          </GlassCard>
        </AnimatedSection>
      </div>

      {/* Table */}
      <AnimatedSection className="mt-6">
        <GlassCard hover={false} className="overflow-hidden">
          <div className="px-5 pt-5 pb-2">
            <h3 className="text-[12px] font-medium uppercase tracking-[0.1em] text-text-tertiary">
              Model Specifications
            </h3>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-[13px]">
              <thead>
                <tr className="border-b border-border text-[11px] uppercase tracking-[0.08em] text-text-tertiary">
                  <th className="px-5 py-3 font-medium">Model</th>
                  <th className="px-5 py-3 font-medium">mAP@50</th>
                  <th className="px-5 py-3 font-medium">Precision</th>
                  <th className="px-5 py-3 font-medium">Recall</th>
                  <th className="px-5 py-3 font-medium">Params</th>
                  <th className="px-5 py-3 font-medium">Latency</th>
                </tr>
              </thead>
              <tbody>
                {modelComparison.map((m, i) => (
                  <motion.tr
                    key={m.model}
                    initial={{ opacity: 0 }}
                    whileInView={{ opacity: 1 }}
                    transition={{ delay: i * 0.04 }}
                    viewport={{ once: true }}
                    className={`border-b border-border transition-colors hover:bg-bg-hover ${
                      m.model === 'Ensemble' ? 'bg-accent-soft' : ''
                    }`}
                  >
                    <td className="px-5 py-3.5 font-medium text-text-primary">{m.model}</td>
                    <td className="px-5 py-3.5 font-mono text-accent">{m.mAP50}%</td>
                    <td className="px-5 py-3.5 font-mono text-text-secondary">{m.precision}%</td>
                    <td className="px-5 py-3.5 font-mono text-text-secondary">{m.recall}%</td>
                    <td className="px-5 py-3.5 text-text-tertiary">{m.params}M</td>
                    <td className="px-5 py-3.5 text-text-tertiary">{m.speed}ms</td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
          </div>
        </GlassCard>
      </AnimatedSection>
    </div>
  )
}
