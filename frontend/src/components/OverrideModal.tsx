import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useAppStore } from "@/lib/store";
import { RUST_API } from "@/lib/config";
import { AlertCircle, X, ShieldAlert } from "lucide-react";

export default function OverrideModal() {
    const { prediction, currentEsi, setEsi } = useAppStore();
    const [isOpen, setIsOpen] = useState(false);
    const [selectedEsi, setSelectedEsi] = useState<number | null>(null);
    const [reason, setReason] = useState("");
    const [status, setStatus] = useState<"idle" | "submitting" | "success" | "error">("idle");

    if (!prediction || !prediction.audit_id) return null;

    const handleSubmit = async () => {
        if (!selectedEsi || !reason.trim()) return;

        setStatus("submitting");
        try {
            const resp = await fetch(`${RUST_API}/audit/override`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    audit_id: prediction.audit_id,
                    override_esi: selectedEsi,
                    reason: reason.trim()
                })
            });

            if (!resp.ok) throw new Error("Failed to submit override");

            setStatus("success");
            
            // Wait a moment before closing to show success state
            setTimeout(() => {
                setIsOpen(false);
                setStatus("idle");
                setReason("");
                setSelectedEsi(null);
                // Force update the UI to the clinician's chosen ESI
                setEsi(selectedEsi as any); 
            }, 1000);

        } catch (err) {
            console.error(err);
            setStatus("error");
        }
    };

    return (
        <>
            {/* The Trigger Button - injected next to Confidence UI */}
            <button 
                onClick={() => setIsOpen(true)}
                className="mt-4 flex items-center gap-2 text-[10px] tracking-widest uppercase font-mono px-4 py-2 rounded-full border border-[var(--color-obsidian-border)] bg-[rgba(255,255,255,0.02)] hover:bg-[rgba(255,255,255,0.05)] text-[var(--color-text-secondary)] transition-colors"
                title="Log a clinician override to the SQLite audit trail"
            >
                <AlertCircle size={12} /> Override AI Decision
            </button>

            {/* The Spring Modal */}
            <AnimatePresence>
                {isOpen && (
                    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
                        <motion.div 
                            initial={{ opacity: 0, scale: 0.95, y: 20 }}
                            animate={{ opacity: 1, scale: 1, y: 0 }}
                            exit={{ opacity: 0, scale: 0.95, y: 20 }}
                            className="w-full max-w-lg glass-panel rounded-2xl overflow-hidden border border-[var(--color-obsidian-border)] shadow-2xl flex flex-col"
                            // Stop propagation so clicking inside doesn't trigger parent handlers
                            onClick={(e) => e.stopPropagation()} 
                        >
                            {/* Header */}
                            <div className="flex justify-between items-center p-4 border-b border-[var(--color-obsidian-border)] bg-[rgba(255,42,42,0.05)]">
                                <h2 className="flex items-center gap-2 text-sm uppercase tracking-widest font-bold text-[#ff2a2a]">
                                    <ShieldAlert size={16} /> Clinician Override
                                </h2>
                                <button onClick={() => setIsOpen(false)} className="text-[var(--color-text-muted)] hover:text-white transition-colors">
                                    <X size={16} />
                                </button>
                            </div>

                            {/* Body */}
                            <div className="p-6 flex flex-col gap-6">
                                <div>
                                    <label className="text-xs uppercase tracking-widest text-[var(--color-text-secondary)] mb-3 block">
                                        Corrected ESI Level
                                    </label>
                                    <div className="flex justify-between gap-2">
                                        {[1, 2, 3, 4, 5].map((level) => (
                                            <button
                                                key={level}
                                                onClick={() => setSelectedEsi(level)}
                                                className={`flex-1 py-3 rounded border font-mono font-bold transition-all ${
                                                    selectedEsi === level 
                                                        ? 'bg-white text-black border-white' 
                                                        : 'bg-[rgba(255,255,255,0.02)] border-[var(--color-obsidian-border)] text-[var(--color-text-secondary)] hover:bg-[rgba(255,255,255,0.08)]'
                                                } ${level === currentEsi ? 'opacity-30 cursor-not-allowed' : ''}`}
                                                disabled={level === currentEsi}
                                                title={level === currentEsi ? "Current AI Prediction" : `Set to ESI ${level}`}
                                            >
                                                {level}
                                            </button>
                                        ))}
                                    </div>
                                </div>

                                <div>
                                    <label className="text-xs uppercase tracking-widest text-[var(--color-text-secondary)] mb-2 block">
                                        Override Justification (Required)
                                    </label>
                                    <textarea 
                                        value={reason}
                                        onChange={(e) => setReason(e.target.value)}
                                        placeholder="e.g. Patient exhibits delayed capillary refill not caught by static vitals..."
                                        className="w-full h-24 bg-[rgba(0,0,0,0.3)] border border-[var(--color-obsidian-border)] rounded-lg p-3 text-sm text-[var(--color-text-primary)] focus:outline-none focus:border-[#ff2a2a] transition-colors resize-none"
                                    />
                                </div>

                                {/* Status & Submit Area */}
                                <div className="flex justify-between items-center mt-2">
                                    <div className="text-xs font-mono text-[var(--color-text-muted)]">
                                        {status === "error" && <span className="text-[#ff2a2a]">Network Error</span>}
                                        {status === "success" && <span className="text-[#00e5ff]">Audit Trail Updated</span>}
                                    </div>
                                    <button 
                                        onClick={handleSubmit}
                                        disabled={!selectedEsi || !reason.trim() || status === "submitting" || status === "success"}
                                        className="px-6 py-2 bg-[#ff2a2a] hover:bg-[#ff4d4d] text-white rounded font-bold tracking-widest text-xs uppercase transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                                    >
                                        {status === "submitting" ? "Committing..." : "Commit Override"}
                                    </button>
                                </div>
                            </div>
                        </motion.div>
                    </div>
                )}
            </AnimatePresence>
        </>
    );
}
