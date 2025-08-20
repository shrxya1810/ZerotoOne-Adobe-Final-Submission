import { useRef, useEffect } from "react";

interface Dot {
  x: number;
  y: number;
  vx: number;
  vy: number;
  radius: number;
  opacity: number;
  connections: number[];
}

export default function ConnectingDotsBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationFrameRef = useRef<number>();
  const dotsRef = useRef<Dot[]>([]);
  const mouseRef = useRef({ x: 0, y: 0 });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Set canvas size
    const resizeCanvas = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };

    resizeCanvas();
    window.addEventListener("resize", resizeCanvas);

    // Initialize dots
    const numDots = 60;
    const maxDistance = 100;

    const initializeDots = () => {
      dotsRef.current = [];
      for (let i = 0; i < numDots; i++) {
        dotsRef.current.push({
          x: Math.random() * canvas.width,
          y: Math.random() * canvas.height,
          vx: (Math.random() - 0.5) * 0.1,
          vy: (Math.random() - 0.5) * 0.1,
          radius: Math.random() * 1 + 0.5,
          opacity: Math.random() * 0.2 + 0.1,
          connections: [],
        });
      }
    };

    initializeDots();

    // Mouse interaction
    const handleMouseMove = (event: MouseEvent) => {
      mouseRef.current.x = event.clientX;
      mouseRef.current.y = event.clientY;
    };

    window.addEventListener("mousemove", handleMouseMove);

    // Animation loop
    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Update dots
      dotsRef.current.forEach((dot, i) => {
        // Move dots
        dot.x += dot.vx;
        dot.y += dot.vy;

        // Bounce off edges
        if (dot.x < 0 || dot.x > canvas.width) dot.vx *= -1;
        if (dot.y < 0 || dot.y > canvas.height) dot.vy *= -1;

        // Keep within bounds
        dot.x = Math.max(0, Math.min(canvas.width, dot.x));
        dot.y = Math.max(0, Math.min(canvas.height, dot.y));

        // Mouse interaction - constellation hover effect
        const mouseDistance = Math.sqrt(
          (dot.x - mouseRef.current.x) ** 2 + (dot.y - mouseRef.current.y) ** 2,
        );

        if (mouseDistance < 150) {
          // Brighten nearby dots like stars
          const hoverIntensity = Math.max(0, 1 - mouseDistance / 150);
          dot.opacity = Math.min(0.8, dot.opacity + hoverIntensity * 0.6);

          // Gentle attraction to mouse
          const angle = Math.atan2(
            dot.y - mouseRef.current.y,
            dot.x - mouseRef.current.x,
          );
          dot.vx += Math.cos(angle) * 0.005;
          dot.vy += Math.sin(angle) * 0.005;
        } else {
          // Fade back to normal
          dot.opacity = Math.max(0.1, dot.opacity - 0.01);
        }

        // Limit velocity for subtle movement
        const maxVel = 0.3;
        if (Math.abs(dot.vx) > maxVel) dot.vx = dot.vx > 0 ? maxVel : -maxVel;
        if (Math.abs(dot.vy) > maxVel) dot.vy = dot.vy > 0 ? maxVel : -maxVel;

        // Find connections
        dot.connections = [];
        dotsRef.current.forEach((otherDot, j) => {
          if (i !== j) {
            const distance = Math.sqrt(
              (dot.x - otherDot.x) ** 2 + (dot.y - otherDot.y) ** 2,
            );
            if (distance < maxDistance) {
              dot.connections.push(j);
            }
          }
        });
      });

      // Draw connections
      dotsRef.current.forEach((dot, i) => {
        dot.connections.forEach((j) => {
          if (i < j) {
            // Avoid drawing duplicate lines
            const otherDot = dotsRef.current[j];
            const distance = Math.sqrt(
              (dot.x - otherDot.x) ** 2 + (dot.y - otherDot.y) ** 2,
            );
            const opacity = 1 - distance / maxDistance;

            if (opacity > 0.05) {
              // Create subtle gradient for constellation lines
              const gradient = ctx.createLinearGradient(
                dot.x,
                dot.y,
                otherDot.x,
                otherDot.y,
              );
              gradient.addColorStop(0, `rgba(255, 215, 0, ${opacity * 0.1})`);
              gradient.addColorStop(0.5, `rgba(255, 193, 7, ${opacity * 0.2})`);
              gradient.addColorStop(1, `rgba(255, 165, 0, ${opacity * 0.1})`);

              ctx.strokeStyle = gradient;
              ctx.lineWidth = opacity * 0.5;
              ctx.beginPath();
              ctx.moveTo(dot.x, dot.y);
              ctx.lineTo(otherDot.x, otherDot.y);
              ctx.stroke();
            }
          }
        });
      });

      // Draw dots
      dotsRef.current.forEach((dot) => {
        // Subtle outer glow for star effect
        const glowGradient = ctx.createRadialGradient(
          dot.x,
          dot.y,
          0,
          dot.x,
          dot.y,
          dot.radius * 6,
        );
        glowGradient.addColorStop(0, `rgba(255, 215, 0, ${dot.opacity * 0.6})`);
        glowGradient.addColorStop(
          0.3,
          `rgba(255, 193, 7, ${dot.opacity * 0.2})`,
        );
        glowGradient.addColorStop(1, "rgba(255, 215, 0, 0)");

        ctx.fillStyle = glowGradient;
        ctx.beginPath();
        ctx.arc(dot.x, dot.y, dot.radius * 4, 0, Math.PI * 2);
        ctx.fill();

        // Inner dot
        const dotGradient = ctx.createRadialGradient(
          dot.x,
          dot.y,
          0,
          dot.x,
          dot.y,
          dot.radius,
        );
        dotGradient.addColorStop(0, `rgba(255, 255, 255, ${dot.opacity})`);
        dotGradient.addColorStop(
          0.7,
          `rgba(255, 215, 0, ${dot.opacity * 0.8})`,
        );
        dotGradient.addColorStop(1, `rgba(255, 165, 0, ${dot.opacity * 0.6})`);

        ctx.fillStyle = dotGradient;
        ctx.beginPath();
        ctx.arc(dot.x, dot.y, dot.radius, 0, Math.PI * 2);
        ctx.fill();
      });

      animationFrameRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      window.removeEventListener("resize", resizeCanvas);
      window.removeEventListener("mousemove", handleMouseMove);
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 z-0 pointer-events-none"
      style={{
        background:
          "linear-gradient(135deg, #000000 0%, #111111 25%, #222222 50%, #111111 75%, #000000 100%)",
      }}
    />
  );
}
