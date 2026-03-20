import { useEffect, useRef } from 'react'

export default function AnimatedGrid() {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    let animationId: number
    let time = 0

    const resize = () => {
      canvas.width = canvas.offsetWidth * 2
      canvas.height = canvas.offsetHeight * 2
      ctx.scale(2, 2)
    }
    resize()
    window.addEventListener('resize', resize)

    const w = () => canvas.width / 2
    const h = () => canvas.height / 2

    const draw = () => {
      time += 0.003
      ctx.clearRect(0, 0, w(), h())

      const cols = 24
      const rows = 16
      const cellW = w() / cols
      const cellH = h() / rows

      // Animated grid lines
      for (let i = 0; i <= cols; i++) {
        const x = i * cellW
        const wave = Math.sin(time * 2 + i * 0.3) * 0.3 + 0.7
        ctx.beginPath()
        ctx.moveTo(x, 0)
        ctx.lineTo(x, h())
        ctx.strokeStyle = `rgba(139, 92, 246, ${0.03 * wave})`
        ctx.lineWidth = 0.5
        ctx.stroke()
      }

      for (let j = 0; j <= rows; j++) {
        const y = j * cellH
        const wave = Math.sin(time * 2 + j * 0.4) * 0.3 + 0.7
        ctx.beginPath()
        ctx.moveTo(0, y)
        ctx.lineTo(w(), y)
        ctx.strokeStyle = `rgba(139, 92, 246, ${0.03 * wave})`
        ctx.lineWidth = 0.5
        ctx.stroke()
      }

      // Floating dots at intersections
      for (let i = 0; i <= cols; i++) {
        for (let j = 0; j <= rows; j++) {
          const x = i * cellW
          const y = j * cellH
          const dist = Math.sqrt(
            Math.pow(x - w() / 2, 2) + Math.pow(y - h() / 2, 2)
          )
          const maxDist = Math.sqrt(Math.pow(w() / 2, 2) + Math.pow(h() / 2, 2))
          const fade = 1 - dist / maxDist
          const pulse = Math.sin(time * 3 + i * 0.5 + j * 0.5) * 0.5 + 0.5

          ctx.beginPath()
          ctx.arc(x, y, 1 + pulse * 0.5, 0, Math.PI * 2)
          ctx.fillStyle = `rgba(139, 92, 246, ${0.08 * fade * pulse})`
          ctx.fill()
        }
      }

      // Scanning line effect
      const scanY = ((Math.sin(time * 0.8) + 1) / 2) * h()
      const gradient = ctx.createLinearGradient(0, scanY - 40, 0, scanY + 40)
      gradient.addColorStop(0, 'rgba(139, 92, 246, 0)')
      gradient.addColorStop(0.5, 'rgba(139, 92, 246, 0.04)')
      gradient.addColorStop(1, 'rgba(139, 92, 246, 0)')
      ctx.fillStyle = gradient
      ctx.fillRect(0, scanY - 40, w(), 80)

      animationId = requestAnimationFrame(draw)
    }

    draw()

    return () => {
      cancelAnimationFrame(animationId)
      window.removeEventListener('resize', resize)
    }
  }, [])

  return (
    <canvas
      ref={canvasRef}
      className="pointer-events-none absolute inset-0 h-full w-full"
      style={{ opacity: 0.7 }}
    />
  )
}
