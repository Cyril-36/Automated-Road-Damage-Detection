import { motion } from 'framer-motion'
import type { ReactNode } from 'react'

interface Props {
  children: ReactNode
  onClick?: () => void
  className?: string
  variant?: 'primary' | 'secondary'
  type?: 'button' | 'submit'
}

export default function GradientButton({
  children,
  onClick,
  className = '',
  variant = 'primary',
  type = 'button',
}: Props) {
  const base =
    variant === 'primary'
      ? 'bg-text-primary text-bg hover:bg-white/90'
      : 'border border-border bg-transparent text-text-secondary hover:text-text-primary hover:border-border-hover'

  return (
    <motion.button
      whileHover={{ scale: 1.015 }}
      whileTap={{ scale: 0.985 }}
      type={type}
      onClick={onClick}
      className={`inline-flex items-center gap-2 rounded-lg px-4 py-2 text-[13px] font-medium transition-all duration-200 ${base} ${className}`}
    >
      {children}
    </motion.button>
  )
}
