"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { useAppStore } from "@/lib/store";
import { cn } from "@/lib/utils";

// Dummy components to be implemented
import TelemetryPane from "@/components/TelemetryPane";
import AICorePane from "@/components/AICorePane";
import RagIntelligencePane from "@/components/RagIntelligencePane";

export default function Home() {
  const currentEsi = useAppStore((state) => state.currentEsi);

  // Mouse tracking for ambient radial gradient
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      setMousePosition({ x: e.clientX, y: e.clientY });
    };
    window.addEventListener("mousemove", handleMouseMove);
    return () => window.removeEventListener("mousemove", handleMouseMove);
  }, []);

  // Determine ambient glow color based on ESI
  const getAmbientColor = () => {
    switch (currentEsi) {
      case 1: return "rgba(255, 42, 42, 0.15)";
      case 2: return "rgba(255, 145, 0, 0.15)";
      case 3: return "rgba(255, 179, 0, 0.15)";
      case 4: return "rgba(0, 229, 255, 0.15)";
      case 5: return "rgba(41, 121, 255, 0.15)";
      default: return "rgba(255, 255, 255, 0.03)";
    }
  };

  return (
    <main className="relative w-screen h-screen overflow-hidden bg-[var(--color-obsidian-bg)] text-[var(--color-text-primary)]">

      {/* Dynamic Ambient Background tracking the cursor */}
      <motion.div
        className="pointer-events-none absolute inset-0 z-0 transition-colors duration-1000"
        animate={{
          background: `radial-gradient(circle 800px at ${mousePosition.x}px ${mousePosition.y}px, ${getAmbientColor()}, transparent)`,
        }}
      />

      {/* Grid Layout Container */}
      <div className="relative z-10 w-full h-full p-4 md:p-6 lg:p-8">
        <div className="w-full h-full grid grid-cols-1 lg:grid-cols-12 gap-6">

          {/* Left Pane: Telemetry (Input) */}
          <div className="lg:col-span-3 h-full flex flex-col glass-panel rounded-2xl overflow-hidden p-6 shadow-2xl">
            <h2 className="text-sm tracking-[0.2em] text-[var(--color-text-secondary)] uppercase mb-6 border-b border-[var(--color-obsidian-border)] pb-4">
              Patient Telemetry
            </h2>
            <TelemetryPane />
          </div>

          {/* Center Pane: AI Core (Hero) */}
          <div className="lg:col-span-6 h-full flex flex-col glass-panel rounded-2xl overflow-hidden p-8 shadow-2xl relative">
            <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-[var(--color-obsidian-border)] to-transparent opacity-50" />
            <AICorePane />
          </div>

          {/* Right Pane: RAG Intelligence */}
          <div className="lg:col-span-3 h-full flex flex-col glass-panel rounded-2xl overflow-hidden p-6 shadow-2xl">
            <h2 className="text-sm tracking-[0.2em] text-[var(--color-text-secondary)] uppercase mb-6 border-b border-[var(--color-obsidian-border)] pb-4">
              Active Intelligence
            </h2>
            <RagIntelligencePane />
          </div>

        </div>
      </div>
    </main>
  );
}
