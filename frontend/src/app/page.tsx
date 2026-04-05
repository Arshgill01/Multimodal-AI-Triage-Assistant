"use client";

import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useAppStore } from "@/lib/store";
import { cn } from "@/lib/utils";

import TelemetryPane from "@/components/TelemetryPane";
import AICorePane from "@/components/AICorePane";
import RagIntelligencePane from "@/components/RagIntelligencePane";
import MCIMode from "@/components/MCIMode";
import { RotateCcw } from "lucide-react";

export default function Home() {
  const currentEsi = useAppStore((state) => state.currentEsi);
  const isMciMode = useAppStore((state) => state.isMciMode);
  const setMciMode = useAppStore((state) => state.setMciMode);
  const reset = useAppStore((state) => state.reset);

  // Mouse tracking for ambient radial gradient
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      setMousePosition({ x: e.clientX, y: e.clientY });
    };
    window.addEventListener("mousemove", handleMouseMove);
    return () => window.removeEventListener("mousemove", handleMouseMove);
  }, []);

  // Determine ambient glow color based on ESI or MCI mode
  const getAmbientColor = () => {
    if (isMciMode) return "rgba(255, 42, 42, 0.08)";
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
      <div className="relative z-10 w-full h-full p-4 md:p-6 lg:p-8 flex flex-col">

        {/* Mode Toggle Bar */}
        <div className="flex items-center justify-between mb-4 shrink-0">
          <div className="flex items-center gap-3">
            <div className="flex bg-[rgba(255,255,255,0.03)] border border-[var(--color-obsidian-border)] rounded-lg overflow-hidden">
              <button
                onClick={() => setMciMode(false)}
                className={`px-4 py-2 text-[10px] font-mono uppercase tracking-widest transition-all ${
                  !isMciMode
                    ? "bg-[rgba(0,229,255,0.1)] text-[#00e5ff] border-r border-[var(--color-obsidian-border)]"
                    : "text-[var(--color-text-muted)] hover:text-[var(--color-text-secondary)] border-r border-[var(--color-obsidian-border)]"
                }`}
              >
                Single Patient
              </button>
              <button
                onClick={() => setMciMode(true)}
                className={`px-4 py-2 text-[10px] font-mono uppercase tracking-widest transition-all ${
                  isMciMode
                    ? "bg-[rgba(255,42,42,0.1)] text-[#ff2a2a]"
                    : "text-[var(--color-text-muted)] hover:text-[var(--color-text-secondary)]"
                }`}
              >
                MCI Mode
              </button>
            </div>
            {isMciMode && (
              <motion.span
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: [0.5, 1, 0.5] }}
                transition={{ duration: 1.5, repeat: Infinity }}
                className="text-[10px] font-mono uppercase tracking-widest text-[#ff2a2a]"
              >
                ● Mass Casualty Active
              </motion.span>
            )}
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={reset}
              className="flex items-center gap-1.5 px-3 py-2 text-[10px] font-mono uppercase tracking-widest text-[var(--color-text-muted)] hover:text-[var(--color-text-secondary)] transition-colors border border-[var(--color-obsidian-border)] rounded-lg bg-[rgba(255,255,255,0.02)] hover:bg-[rgba(255,255,255,0.05)]"
              title="Clear all data and start fresh"
            >
              <RotateCcw size={10} /> New Patient
            </button>
            <span className="text-[10px] font-mono text-[var(--color-text-muted)] tracking-widest uppercase">
              Frostbyte Obsidian HUD v2
            </span>
          </div>
        </div>

        {/* Conditional Layout */}
        <AnimatePresence mode="wait">
          {isMciMode ? (
            <motion.div
              key="mci"
              initial={{ opacity: 0, scale: 0.98 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.98 }}
              transition={{ duration: 0.3 }}
              className="flex-1 glass-panel rounded-2xl overflow-hidden p-6 shadow-2xl"
            >
              <MCIMode />
            </motion.div>
          ) : (
            <motion.div
              key="single"
              initial={{ opacity: 0, scale: 0.98 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.98 }}
              transition={{ duration: 0.3 }}
              className="flex-1 min-h-0 grid grid-cols-1 lg:grid-cols-12 gap-6"
            >
              {/* Left Pane: Telemetry (Input) */}
              <div className="lg:col-span-3 min-h-0 flex flex-col glass-panel rounded-2xl overflow-hidden p-6 shadow-2xl">
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
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </main>
  );
}
