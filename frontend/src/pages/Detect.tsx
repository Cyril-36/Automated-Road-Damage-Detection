import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Upload,
  Image as ImageIcon,
  Loader2,
  AlertTriangle,
  CheckCircle2,
  X,
  ChevronDown,
} from 'lucide-react'
import GradientButton from '../components/GradientButton'
import GlassCard from '../components/GlassCard'
import AnimatedSection from '../components/AnimatedSection'

type DetectionMode = 'fast' | 'balanced' | 'accurate'

interface Detection {
  class: string
  label: string
  confidence: number
  bbox: [number, number, number, number]
}

interface Result {
  detections: Detection[]
  count: number
  image_size: [number, number]
  mode: string
  annotated_image?: string
}

const CLASS_COLORS: Record<string, string> = {
  D00: '#f87171',
  D10: '#fb923c',
  D20: '#fbbf24',
  D40: '#60a5fa',
  D50: '#a78bfa',
}

const CLASS_LABELS: Record<string, string> = {
  D00: 'Longitudinal Crack',
  D10: 'Transverse Crack',
  D20: 'Alligator Crack',
  D40: 'Pothole',
  D50: 'Other Corruption',
}

const MODES: { value: DetectionMode; label: string; desc: string }[] = [
  { value: 'fast', label: 'Fast', desc: '1 model ~20ms' },
  { value: 'balanced', label: 'Balanced', desc: '2 models ~100ms' },
  { value: 'accurate', label: 'Accurate', desc: '3 models + WBF ~150ms' },
]

export default function Detect() {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [mode, setMode] = useState<DetectionMode>('accurate')
  const [confidence, setConfidence] = useState(0.25)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<Result | null>(null)
  const [error, setError] = useState<string | null>(null)

  const onDrop = useCallback((accepted: File[]) => {
    const f = accepted[0]
    if (!f) return
    setFile(f)
    setPreview(URL.createObjectURL(f))
    setResult(null)
    setError(null)
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.jpg', '.jpeg', '.png', '.bmp'] },
    maxFiles: 1,
  })

  const handleDetect = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const formData = new FormData()
      formData.append('image', file)
      formData.append('mode', mode)
      formData.append('confidence', confidence.toString())

      const res = await fetch('/api/predict', { method: 'POST', body: formData })
      const data = await res.json()
      if (!res.ok) throw new Error(data.error || `Server error: ${res.status}`)
      setResult(data as Result)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Detection failed')
    } finally {
      setLoading(false)
    }
  }

  const clearImage = () => {
    setFile(null)
    setPreview(null)
    setResult(null)
    setError(null)
  }

  return (
    <div className="mx-auto max-w-5xl px-6 pt-28 pb-20">
      <AnimatedSection className="mb-10">
        <p className="text-[12px] font-medium uppercase tracking-[0.15em] text-accent">Detection</p>
        <h1 className="mt-2 text-2xl font-semibold tracking-[-0.025em] text-text-primary">
          Analyze road damage
        </h1>
        <p className="mt-2 text-[14px] text-text-secondary">
          Upload an image to detect and classify surface damage.
        </p>
      </AnimatedSection>

      <div className="grid gap-6 lg:grid-cols-5">
        {/* Left */}
        <div className="flex flex-col gap-4 lg:col-span-3">
          {/* Dropzone */}
          <GlassCard hover={false} className="overflow-hidden">
            {!preview ? (
              <div
                {...getRootProps()}
                className={`flex cursor-pointer flex-col items-center justify-center gap-4 py-24 px-8 transition-colors duration-200 ${
                  isDragActive ? 'bg-accent-soft' : 'hover:bg-bg-hover'
                }`}
              >
                <input {...getInputProps()} />
                <motion.div
                  animate={isDragActive ? { scale: 1.05 } : { scale: 1 }}
                  transition={{ duration: 0.2 }}
                  className="flex h-12 w-12 items-center justify-center rounded-xl border border-border bg-bg-subtle text-text-tertiary"
                >
                  <Upload className="h-5 w-5" />
                </motion.div>
                <div className="text-center">
                  <p className="text-[13px] font-medium text-text-primary">
                    {isDragActive ? 'Drop here' : 'Drop a road image'}
                  </p>
                  <p className="mt-1 text-[12px] text-text-tertiary">or click to browse</p>
                </div>
              </div>
            ) : (
              <div className="relative group">
                <button
                  onClick={clearImage}
                  className="absolute right-3 top-3 z-10 rounded-md bg-bg/80 backdrop-blur p-1.5 text-text-tertiary opacity-0 group-hover:opacity-100 transition-opacity hover:text-text-primary"
                >
                  <X className="h-3.5 w-3.5" />
                </button>
                <img
                  src={
                    result?.annotated_image
                      ? `data:image/jpeg;base64,${result.annotated_image}`
                      : preview
                  }
                  alt="Road"
                  className="w-full object-contain"
                  style={{ maxHeight: 420 }}
                />
              </div>
            )}
          </GlassCard>

          {/* Controls */}
          <GlassCard hover={false} className="p-5">
            <div className="grid gap-5 sm:grid-cols-2">
              <div>
                <label className="mb-2 block text-[11px] font-medium uppercase tracking-[0.1em] text-text-tertiary">
                  Mode
                </label>
                <div className="relative">
                  <select
                    value={mode}
                    onChange={(e) => setMode(e.target.value as DetectionMode)}
                    className="w-full appearance-none rounded-lg border border-border bg-bg px-3 py-2.5 pr-8 text-[13px] text-text-primary outline-none transition-colors focus:border-accent/40"
                  >
                    {MODES.map((m) => (
                      <option key={m.value} value={m.value}>
                        {m.label} — {m.desc}
                      </option>
                    ))}
                  </select>
                  <ChevronDown className="pointer-events-none absolute right-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-text-tertiary" />
                </div>
              </div>

              <div>
                <label className="mb-2 flex items-center justify-between text-[11px] font-medium uppercase tracking-[0.1em] text-text-tertiary">
                  <span>Confidence</span>
                  <span className="font-mono text-text-secondary">{(confidence * 100).toFixed(0)}%</span>
                </label>
                <input
                  type="range"
                  min={0.1}
                  max={0.9}
                  step={0.05}
                  value={confidence}
                  onChange={(e) => setConfidence(parseFloat(e.target.value))}
                  className="mt-3 w-full"
                />
              </div>
            </div>

            <div className="mt-5">
              <GradientButton
                onClick={handleDetect}
                className={`w-full justify-center ${!file || loading ? 'pointer-events-none opacity-40' : ''}`}
              >
                {loading ? (
                  <><Loader2 className="h-3.5 w-3.5 animate-spin" /> Analyzing...</>
                ) : (
                  <><ImageIcon className="h-3.5 w-3.5" /> Detect Damage</>
                )}
              </GradientButton>
            </div>
          </GlassCard>
        </div>

        {/* Right — Results */}
        <div className="lg:col-span-2">
          <AnimatePresence mode="wait">
            {error && (
              <motion.div
                key="error"
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.3 }}
              >
                <GlassCard hover={false} className="p-5">
                  <div className="flex items-start gap-3">
                    <AlertTriangle className="h-4 w-4 shrink-0 text-danger mt-0.5" />
                    <div>
                      <p className="text-[13px] font-medium text-text-primary">Detection failed</p>
                      <p className="mt-1 text-[12px] text-text-tertiary">{error}</p>
                      <p className="mt-2 text-[11px] text-text-tertiary">
                        Ensure the Flask API is running on port 8080.
                      </p>
                    </div>
                  </div>
                </GlassCard>
              </motion.div>
            )}

            {result && (
              <motion.div
                key="result"
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.3 }}
                className="flex flex-col gap-3"
              >
                <GlassCard hover={false} className="p-5">
                  <div className="flex items-center gap-2 text-success mb-4">
                    <CheckCircle2 className="h-3.5 w-3.5" />
                    <span className="text-[12px] font-medium">Analysis complete</span>
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <div className="rounded-lg bg-bg-subtle p-3.5 text-center">
                      <p className="text-xl font-semibold text-text-primary">{result.count}</p>
                      <p className="mt-0.5 text-[11px] text-text-tertiary">Damages</p>
                    </div>
                    <div className="rounded-lg bg-bg-subtle p-3.5 text-center">
                      <p className="text-xl font-semibold capitalize text-text-primary">{result.mode}</p>
                      <p className="mt-0.5 text-[11px] text-text-tertiary">Mode</p>
                    </div>
                  </div>
                </GlassCard>

                {result.detections.length > 0 ? (
                  <GlassCard hover={false} className="p-5">
                    <h3 className="mb-3 text-[12px] font-medium uppercase tracking-[0.1em] text-text-tertiary">
                      Detections
                    </h3>
                    <div className="flex flex-col gap-2">
                      {result.detections.map((d, i) => (
                        <motion.div
                          key={i}
                          initial={{ opacity: 0, x: -8 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: i * 0.04, duration: 0.3 }}
                          className="flex items-center gap-3 rounded-lg bg-bg-subtle p-3"
                        >
                          <div
                            className="h-2 w-2 rounded-full shrink-0"
                            style={{ backgroundColor: CLASS_COLORS[d.class] || '#71717a' }}
                          />
                          <div className="flex-1 min-w-0">
                            <p className="text-[13px] font-medium text-text-primary truncate">
                              {d.label || CLASS_LABELS[d.class] || d.class}
                            </p>
                            <p className="text-[11px] text-text-tertiary">{d.class}</p>
                          </div>
                          <span className="font-mono text-[12px] font-medium text-text-secondary">
                            {(d.confidence * 100).toFixed(1)}%
                          </span>
                        </motion.div>
                      ))}
                    </div>
                  </GlassCard>
                ) : (
                  <GlassCard hover={false} className="p-8 text-center">
                    <CheckCircle2 className="mx-auto h-6 w-6 text-success" />
                    <p className="mt-3 text-[13px] font-medium text-text-primary">No damage detected</p>
                    <p className="mt-1 text-[12px] text-text-tertiary">Road appears in good condition.</p>
                  </GlassCard>
                )}
              </motion.div>
            )}

            {!result && !error && (
              <motion.div
                key="empty"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                <GlassCard hover={false} className="flex flex-col items-center gap-3 py-20 text-center">
                  <div className="flex h-10 w-10 items-center justify-center rounded-lg border border-border bg-bg-subtle text-text-tertiary">
                    <ImageIcon className="h-4 w-4" />
                  </div>
                  <p className="text-[13px] text-text-tertiary">Results will appear here</p>
                </GlassCard>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  )
}
