import { motion } from 'framer-motion'
import type { ReactNode } from 'react'

interface Props {
  children: ReactNode
  className?: string
  hover?: boolean
}

export default function GlassCard({ children, className = '', hover = true }: Props) {
  return (
    <motion.div
      whileHover={hover ? { y: -2 } : undefined}
      transition={{ duration: 0.3, ease: [0.25, 0.1, 0, 1] }}
      className={`rounded-xl border border-border bg-bg-elevated ${className}`}
    >
      {children}
    </motion.div>
  )
}
