import { useEffect, useState, useRef } from 'react'
import { useInView } from 'framer-motion'

interface Props {
  end: number
  duration?: number
  suffix?: string
  decimals?: number
}

export default function CountUp({ end, duration = 1800, suffix = '', decimals = 0 }: Props) {
  const [count, setCount] = useState(0)
  const ref = useRef<HTMLSpanElement>(null)
  const isInView = useInView(ref, { once: true })

  useEffect(() => {
    if (!isInView) return
    const startTime = performance.now()

    function animate(now: number) {
      const progress = Math.min((now - startTime) / duration, 1)
      const eased = 1 - Math.pow(1 - progress, 4)
      setCount(eased * end)
      if (progress < 1) requestAnimationFrame(animate)
    }

    requestAnimationFrame(animate)
  }, [isInView, end, duration])

  return (
    <span ref={ref}>
      {count.toFixed(decimals)}{suffix}
    </span>
  )
}
