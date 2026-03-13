import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useAppStore } from "@/lib/store";
import ShapWaterfall from "./ShapWaterfall";
import OverrideModal from "./OverrideModal";

const SEQUENCE_PHASES = {
    idle: "Awaiting Telemetry...",
    extracting: "[SYS] Extracting Embeddings...",
    routing: "[SYS] Rust FFI Initialized. Routing vectors...",
    inferring: "[SYS] LightGBM Engine: Inferring ESI...",
    explainability: "[SYS] TreeExplainer: Computing SHAP values...",
    rag: "[SYS] Triage complete. Syncing intelligence...",
    complete: "Analysis Complete",
};

export default function AICorePane() {
    const { analysisPhase, prediction, currentEsi } = useAppStore();

    const getEsiColor = () => {
        switch (currentEsi) {
            case 1: return "text-[#ff2a2a]";
            case 2: return "text-[#ff9100]";
            case 3: return "text-[#ffb300]";
            case 4: return "text-[#00e5ff]";
            case 5: return "text-[#2979ff]";
            default: return "text-[var(--color-text-muted)]";
        }
    };

    return (
        <div className="w-full h-full flex flex-col items-center justify-between py-4">

            {/* Sequence Roller Tracker */}
            <div className="h-8 overflow-hidden w-full text-center relative border-b border-[rgba(255,255,255,0.05)] pb-6 mb-8 flex justify-center">
                <AnimatePresence mode="popLayout">
                    <motion.p
                        key={analysisPhase}
                        initial={{ y: 20, opacity: 0 }}
                        animate={{ y: 0, opacity: 1 }}
                        exit={{ y: -20, opacity: 0 }}
                        transition={{ type: "spring", stiffness: 300, damping: 30 }}
                        className={`font-mono text-sm uppercase tracking-widest ${analysisPhase === 'idle' ? 'text-[var(--color-text-muted)]' : 'text-[#00e5ff] animate-pulse'
                            }`}
                    >
                        {SEQUENCE_PHASES[analysisPhase]}
                    </motion.p>
                </AnimatePresence>
            </div>

            {/* Massive ESI Hero Display */}
            <div className="flex-1 flex flex-col items-center justify-center w-full min-h-[250px] relative">
                <AnimatePresence mode="wait">
                    {!prediction ? (
                        <motion.div
                            key="empty"
                            initial={{ opacity: 0, scale: 0.9 }}
                            animate={{ opacity: 1, scale: 1 }}
                            exit={{ opacity: 0, scale: 1.1, filter: "blur(10px)" }}
                            className="text-center"
                        >
                            <div className="w-48 h-48 border border-[var(--color-obsidian-border)] rounded-full flex items-center justify-center relative">
                                <div className="absolute inset-0 border border-[var(--color-text-muted)] opacity-20 rounded-full animate-[spin_10s_linear_infinite]" />
                                <div className="absolute inset-4 border border-[var(--color-text-muted)] opacity-10 rounded-full animate-[spin_15s_linear_infinite_reverse]" />
                                <span className="text-[var(--color-text-muted)] font-mono text-xs uppercase tracking-widest">No Data</span>
                            </div>
                        </motion.div>
                    ) : (
                        <motion.div
                            key="result"
                            initial={{ opacity: 0, scale: 0.5, filter: "blur(20px)" }}
                            animate={{ opacity: 1, scale: 1, filter: "blur(0px)" }}
                            transition={{ type: "spring", stiffness: 200, damping: 20 }}
                            className="text-center w-full flex flex-col items-center gap-6"
                        >
                            <div className="flex flex-col items-center justify-center">
                                <motion.span
                                    initial={{ y: -20, opacity: 0 }}
                                    animate={{ y: 0, opacity: 1 }}
                                    transition={{ delay: 0.3 }}
                                    className="text-sm font-mono tracking-[0.3em] uppercase text-[var(--color-text-secondary)]"
                                >
                                    Predicted ESI Level
                                </motion.span>
                                <motion.div
                                    className={`text-[12rem] leading-none font-bold tracking-tighter ${getEsiColor()} drop-shadow-[0_0_30px_currentColor]`}
                                    initial={{ scale: 0.5 }}
                                    animate={{ scale: 1 }}
                                    transition={{ type: "spring", bounce: 0.5 }}
                                >
                                    {prediction.predicted_esi}
                                </motion.div>
                                <motion.div
                                    initial={{ opacity: 0 }}
                                    animate={{ opacity: 1 }}
                                    transition={{ delay: 0.6 }}
                                    className="uppercase tracking-widest mt-2 font-bold text-lg"
                                >
                                    {prediction.esi_label}
                                </motion.div>
                            </div>

                            {/* Confidence Metrics Badge */}
                            {prediction.confidence && (
                                <motion.div
                                    initial={{ opacity: 0, y: 10 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    transition={{ delay: 0.8 }}
                                    className={`px-4 py-2 rounded-full border ${prediction.confidence.is_uncertain
                                            ? 'border-[#ffb300] bg-[rgba(255,179,0,0.1)] text-[#ffb300]'
                                            : 'border-[var(--color-obsidian-border)] bg-[rgba(255,255,255,0.03)] text-[var(--color-text-secondary)]'
                                        } text-xs font-mono flex gap-4`}
                                >
                                    <span>Confidence: {(prediction.confidence.top_probability * 100).toFixed(1)}%</span>
                                    {prediction.confidence.is_uncertain && <span>! MANUAL REVIEW RECOMMENDED</span>}
                                </motion.div>
                            )}
                            
                            {/* Human-in-the-Loop Override */}
                            <OverrideModal />
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>

            {/* SHAP Waterfall Chart (Appears after prediction) */}
            <div className="h-[40%] w-full border-t border-[var(--color-obsidian-border)] pt-6 mt-4 relative">
                {!prediction ? (
                    <div className="absolute inset-0 flex items-center justify-center">
                        <span className="text-[var(--color-text-muted)] font-mono text-xs tracking-widest uppercase">Shapley Additive Explanations</span>
                    </div>
                ) : (
                    <ShapWaterfall />
                )}
            </div>

        </div>
    );
}
