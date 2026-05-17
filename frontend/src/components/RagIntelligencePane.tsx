import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useAppStore } from "@/lib/store";
import type { SimilarCaseEvidence } from "@/lib/api-types";
import { PYTHON_API } from "@/lib/config";
import ReactMarkdown from "react-markdown";
import { BrainCircuit, FileClock, Activity, HeartPulse, AlertTriangle } from "lucide-react";

export default function RagIntelligencePane() {
    const { prediction, ragStream, similarCases, analysisPhase, patientData, appendRagStream, setSimilarCases } = useAppStore();
    const [isStreaming, setIsStreaming] = useState(false);

    // Trigger RAG stream when ESI prediction completes
    useEffect(() => {
        if (analysisPhase === "rag" && prediction) {
            // Reset RAG state
            useAppStore.setState({ ragStream: "", similarCases: [] });
            setIsStreaming(true);

            const { chief_complaint, age, heart_rate, resp_rate, spo2, temp_f, systolic_bp, pain_scale } = patientData;
            const requestBody = {
                complaint: chief_complaint,
                vitals: { age, heart_rate, resp_rate, spo2, temp_f, systolic_bp, pain_scale },
                predicted_esi: prediction.predicted_esi,
            };

            const initStreaming = async () => {
                try {
                    const response = await fetch(`${PYTHON_API}/rag-stream`, {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify(requestBody),
                    });

                    if (!response.body) throw new Error("No readable stream available");

                    const reader = response.body.getReader();
                    const decoder = new TextDecoder();

                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;

                        const chunk = decoder.decode(value, { stream: true });
                        const lines = chunk.split("\n");

                        for (const line of lines) {
                            if (line.startsWith("data: ")) {
                                const dataString = line.substring(6);
                                if (dataString === "[DONE]") {
                                    setIsStreaming(false);
                                    break;
                                }

                                try {
                                    const parsed = JSON.parse(dataString);
                                    if (parsed.similar_cases) {
                                        setSimilarCases(parsed.similar_cases);
                                    } else if (parsed.token) {
                                        appendRagStream(parsed.token);
                                    }
                                } catch (e) {
                                    // ignoring incomplete JSON chunks sometimes emitted by SSE
                                }
                            }
                        }
                    }
                } catch (err) {
                    console.error("RAG Stream Error:", err);
                    setIsStreaming(false);
                }
            };

            initStreaming();
        }
    }, [analysisPhase, prediction]);

    // If no analysis is happening, show idle state
    if (!prediction && analysisPhase === "idle") {
        return (
            <div className="w-full h-full flex flex-col items-center justify-center text-center opacity-50">
                <BrainCircuit size={48} className="mb-4 text-[var(--color-obsidian-border)]" />
                <p className="font-mono text-[var(--color-text-secondary)] text-sm uppercase tracking-widest">
                    Intelligence Engine Offline
                </p>
                <p className="text-[10px] text-[var(--color-text-muted)] mt-2 tracking-widest uppercase">
                    Awaiting ESI determination for sequence initialization.
                </p>
            </div>
        );
    }

    return (
        <div className="flex flex-col h-full gap-4">
            {/* Similar Cases Deck */}
            <div className="h-1/3 flex flex-col border-b border-[var(--color-obsidian-border)] pb-4">
                <h3 className="text-[10px] tracking-widest uppercase text-[var(--color-text-secondary)] mb-3 flex items-center gap-2">
                    <Activity size={12} className="text-[#00e5ff]" /> Hybrid Evidence Retrieval
                </h3>

                <div className="flex-1 overflow-y-auto pr-2 flex flex-col gap-2">
                    <AnimatePresence>
                        {similarCases.length === 0 && isStreaming ? (
                            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="text-xs text-[var(--color-text-muted)] animate-pulse">
                                Querying Vector Store...
                            </motion.div>
                        ) : (
                            similarCases.map((c: SimilarCaseEvidence, i) => (
                                <motion.div
                                    key={i}
                                    initial={{ opacity: 0, x: 20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    transition={{ delay: i * 0.1 }}
                                    className="p-3 bg-[rgba(255,255,255,0.02)] border border-[var(--color-obsidian-border)] rounded hover:bg-[rgba(255,255,255,0.05)] transition-colors text-xs"
                                >
                                    <div className="flex justify-between items-start mb-1">
                                        <div className="flex items-center gap-2">
                                            <span className={`font-bold ${c.target_esi === 1 ? 'text-[#ff2a2a]' : c.target_esi === 2 ? 'text-[#ff9100]' : 'text-[#ffb300]'}`}>
                                                ESI {c.target_esi}
                                            </span>
                                            {c.flag_high_risk === 1 && (
                                                <AlertTriangle size={12} className="text-[#ff2a2a]" />
                                            )}
                                        </div>
                                        <span className="text-[var(--color-text-muted)] font-mono">{(c.similarity * 100).toFixed(0)}%</span>
                                    </div>
                                    <p className="text-[var(--color-text-primary)] line-clamp-2 italic mb-2">&ldquo;{c.complaint}&rdquo;</p>
                                    <div className="flex items-center gap-3 text-[10px] text-[var(--color-text-muted)] font-mono">
                                        {c.source === "both" ? (
                                            <span className="flex items-center gap-1 text-[#00e5ff]">
                                                <Activity size={10} /> DUAL
                                            </span>
                                        ) : c.source === "vitals" ? (
                                            <span className="flex items-center gap-1 text-[#ff9100]">
                                                <HeartPulse size={10} /> VITALS
                                            </span>
                                        ) : (
                                            <span className="flex items-center gap-1 text-[#a78bfa]">
                                                <FileClock size={10} /> TEXT
                                            </span>
                                        )}
                                        <span>
                                            txt:{(c.text_similarity ? c.text_similarity * 100 : 0).toFixed(0)}%
                                        </span>
                                        <span>
                                            vit:{(c.vitals_similarity ? c.vitals_similarity * 100 : 0).toFixed(0)}%
                                        </span>
                                    </div>
                                </motion.div>
                            ))
                        )}
                    </AnimatePresence>
                </div>
            </div>

            {/* RAG Markdown Stream */}
            <div className="flex-[2] flex flex-col min-h-0 relative">
                <h3 className="text-[10px] tracking-widest uppercase text-[var(--color-text-secondary)] mb-2 flex justify-between items-center shrink-0">
                    <div className="flex items-center gap-2">
                        <BrainCircuit size={12} /> Clinical Action Plan
                    </div>
                    {isStreaming && <span className="text-[#00e5ff] animate-pulse">Streaming...</span>}
                </h3>

                <div className="flex-1 overflow-y-auto pr-3 text-sm leading-relaxed
                                prose prose-invert max-w-none 
                                prose-p:text-[var(--color-text-secondary)] prose-p:my-2
                                prose-ul:my-2 prose-ul:pl-4 prose-li:my-1
                                prose-headings:text-[var(--color-text-primary)] prose-headings:font-bold prose-headings:text-sm prose-headings:mt-4 prose-headings:mb-2
                                prose-strong:text-[#00e5ff] prose-li:text-[var(--color-text-secondary)] prose-hr:border-[var(--color-obsidian-border)] prose-hr:my-4">
                    <ReactMarkdown>{ragStream}</ReactMarkdown>
                    {isStreaming && <span className="inline-block w-2 h-4 ml-1 mt-1 align-middle bg-[#00e5ff] animate-[pulse_1s_infinite]" />}
                </div>
            </div>
        </div>
    );
}
