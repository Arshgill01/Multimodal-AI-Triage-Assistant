"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useAppStore } from "@/lib/store";
import { RUST_API } from "@/lib/config";
import { Zap, AlertTriangle, Loader2 } from "lucide-react";

// ── Simulated patient generator ─────────────────────────────
const COMPLAINTS = [
    "Chest pain radiating to left arm", "Severe shortness of breath",
    "Altered mental status", "Fall from standing height", "Abdominal pain x 3 days",
    "Laceration to forearm", "Headache and vomiting", "High fever and rash",
    "Motor vehicle accident", "Difficulty breathing, history of COPD",
    "Syncope episode", "Allergic reaction with swelling", "Nausea and diarrhea",
    "Lower back pain", "Sprained ankle", "Sore throat x 5 days",
    "Seizure witnessed", "Suicidal ideation", "Burn to hand",
    "Eye pain and redness", "Flank pain with hematuria", "Palpitations",
    "Diabetic with blood sugar 450", "Stroke symptoms, left-sided weakness",
    "Gunshot wound to abdomen",
];

function randomBetween(min: number, max: number) {
    return Math.round((Math.random() * (max - min) + min) * 10) / 10;
}

function generatePatients(count: number) {
    return Array.from({ length: count }, (_, i) => ({
        age: Math.round(randomBetween(18, 90)),
        heart_rate: Math.round(randomBetween(40, 180)),
        resp_rate: Math.round(randomBetween(8, 40)),
        spo2: Math.round(randomBetween(75, 100)),
        temp_f: randomBetween(95, 104),
        systolic_bp: Math.round(randomBetween(70, 200)),
        pain_scale: Math.round(randomBetween(0, 10)),
        chief_complaint: COMPLAINTS[i % COMPLAINTS.length],
    }));
}

// ── ESI color map ───────────────────────────────────────────
const ESI_COLORS: Record<number, string> = {
    1: "#ff2a2a",
    2: "#ff9100",
    3: "#ffb300",
    4: "#00e5ff",
    5: "#2979ff",
};

const ESI_LABELS: Record<number, string> = {
    1: "Resuscitation",
    2: "Emergent",
    3: "Urgent",
    4: "Less Urgent",
    5: "Non-Urgent",
};

export default function MCIMode() {
    const { batchResults, setBatchResults } = useAppStore();
    const [isLoading, setIsLoading] = useState(false);
    const [elapsed, setElapsed] = useState<number | null>(null);
    const [patientCount, setPatientCount] = useState(50);

    const runBatchTriage = async () => {
        setIsLoading(true);
        setBatchResults([]);
        setElapsed(null);

        const patients = generatePatients(patientCount);
        const start = performance.now();

        try {
            const resp = await fetch(`${RUST_API}/batch-predict`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ patients }),
            });

            if (!resp.ok) throw new Error("Batch predict failed");

            const data = await resp.json();
            const end = performance.now();
            setElapsed(Math.round(end - start));
            setBatchResults(data.patients || []);
        } catch (err) {
            console.error(err);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="w-full h-full flex flex-col">
            {/* MCI Header Bar */}
            <div className="flex items-center justify-between mb-6 pb-4 border-b border-[var(--color-obsidian-border)]">
                <div className="flex items-center gap-3">
                    <AlertTriangle className="text-[#ff2a2a]" size={20} />
                    <h1 className="text-lg font-bold tracking-widest uppercase text-[#ff2a2a]">
                        Mass Casualty Triage
                    </h1>
                </div>

                <div className="flex items-center gap-4">
                    <div className="flex items-center gap-2">
                        <label className="text-xs text-[var(--color-text-muted)] uppercase tracking-widest">Simulated:</label>
                        <select
                            value={patientCount}
                            onChange={(e) => setPatientCount(parseInt(e.target.value))}
                            className="bg-[rgba(0,0,0,0.4)] border border-[var(--color-obsidian-border)] rounded px-2 py-1 text-xs font-mono text-[var(--color-text-primary)]"
                        >
                            <option value={10}>10</option>
                            <option value={25}>25</option>
                            <option value={50}>50</option>
                            <option value={100}>100</option>
                        </select>
                    </div>

                    <button
                        onClick={runBatchTriage}
                        disabled={isLoading}
                        className="px-5 py-2 bg-[#ff2a2a] hover:bg-[#ff4d4d] text-white rounded font-bold text-xs uppercase tracking-widest transition-colors disabled:opacity-50 flex items-center gap-2"
                    >
                        {isLoading ? (
                            <><Loader2 size={14} className="animate-spin" /> Processing...</>
                        ) : (
                            <><Zap size={14} /> Execute Batch Triage</>
                        )}
                    </button>
                </div>
            </div>

            {/* Performance Stats */}
            {elapsed !== null && (
                <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex gap-6 mb-4 text-xs font-mono"
                >
                    <span className="text-[var(--color-text-muted)]">
                        Processed <span className="text-[var(--color-text-primary)] font-bold">{batchResults.length}</span> patients
                    </span>
                    <span className="text-[var(--color-text-muted)]">
                        Latency: <span className="text-[#00e5ff] font-bold">{elapsed}ms</span>
                    </span>
                    <span className="text-[var(--color-text-muted)]">
                        Throughput: <span className="text-[#00c853] font-bold">{elapsed > 0 ? Math.round((batchResults.length / elapsed) * 1000) : "∞"}</span> patients/sec
                    </span>
                </motion.div>
            )}

            {/* ESI Distribution Bar */}
            {batchResults.length > 0 && (
                <motion.div
                    initial={{ opacity: 0, scaleX: 0 }}
                    animate={{ opacity: 1, scaleX: 1 }}
                    className="flex h-2 rounded-full overflow-hidden mb-4 origin-left"
                >
                    {[1, 2, 3, 4, 5].map((esi) => {
                        const count = batchResults.filter((r: any) => r.prediction.predicted_esi === esi).length;
                        const pct = (count / batchResults.length) * 100;
                        if (pct === 0) return null;
                        return (
                            <div
                                key={esi}
                                style={{ width: `${pct}%`, backgroundColor: ESI_COLORS[esi] }}
                                title={`ESI ${esi}: ${count} patients (${pct.toFixed(0)}%)`}
                            />
                        );
                    })}
                </motion.div>
            )}

            {/* The Dense Triage Table */}
            <div className="flex-1 overflow-auto rounded-lg border border-[var(--color-obsidian-border)]">
                <table className="w-full text-xs font-mono">
                    <thead className="sticky top-0 bg-[rgba(7,7,9,0.95)] border-b border-[var(--color-obsidian-border)]">
                        <tr className="text-[var(--color-text-muted)] uppercase tracking-widest">
                            <th className="text-left px-3 py-2">#</th>
                            <th className="text-left px-3 py-2">ESI</th>
                            <th className="text-left px-3 py-2">Chief Complaint</th>
                            <th className="text-right px-3 py-2">HR</th>
                            <th className="text-right px-3 py-2">SpO2</th>
                            <th className="text-right px-3 py-2">BP</th>
                            <th className="text-right px-3 py-2">Temp</th>
                            <th className="text-right px-3 py-2">Confidence</th>
                            <th className="text-center px-3 py-2">Flag</th>
                        </tr>
                    </thead>
                    <tbody>
                        <AnimatePresence>
                            {batchResults.map((result: any, i: number) => {
                                const p = result.prediction;
                                const esiColor = ESI_COLORS[p.predicted_esi] || "#888";
                                const conf = p.confidence?.top_probability ?? 0;
                                return (
                                    <motion.tr
                                        key={`${result.index}-${i}`}
                                        initial={{ opacity: 0, x: -20 }}
                                        animate={{ opacity: 1, x: 0 }}
                                        transition={{ delay: i * 0.015 }}
                                        className="border-b border-[rgba(255,255,255,0.03)] hover:bg-[rgba(255,255,255,0.03)] transition-colors"
                                    >
                                        <td className="px-3 py-2 text-[var(--color-text-muted)]">{i + 1}</td>
                                        <td className="px-3 py-2">
                                            <span
                                                className="inline-flex items-center justify-center w-6 h-6 rounded font-bold text-sm"
                                                style={{ color: esiColor, border: `1px solid ${esiColor}`, backgroundColor: `${esiColor}15` }}
                                            >
                                                {p.predicted_esi}
                                            </span>
                                        </td>
                                        <td className="px-3 py-2 text-[var(--color-text-primary)] max-w-[240px] truncate">
                                            {result.chief_complaint}
                                        </td>
                                        <td className="px-3 py-2 text-right text-[var(--color-text-secondary)]">
                                            {/* Use inline data from the feature vector: index 1 = heart_rate */}
                                            {p.feature_vector?.[1]?.toFixed(0) ?? "—"}
                                        </td>
                                        <td className="px-3 py-2 text-right text-[var(--color-text-secondary)]">
                                            {p.feature_vector?.[3]?.toFixed(0) ?? "—"}
                                        </td>
                                        <td className="px-3 py-2 text-right text-[var(--color-text-secondary)]">
                                            {p.feature_vector?.[5]?.toFixed(0) ?? "—"}
                                        </td>
                                        <td className="px-3 py-2 text-right text-[var(--color-text-secondary)]">
                                            {p.feature_vector?.[4]?.toFixed(1) ?? "—"}
                                        </td>
                                        <td className="px-3 py-2 text-right">
                                            <span style={{ color: conf > 0.8 ? "#00c853" : conf > 0.5 ? "#ffb300" : "#ff2a2a" }}>
                                                {(conf * 100).toFixed(1)}%
                                            </span>
                                        </td>
                                        <td className="px-3 py-2 text-center">
                                            {p.confidence?.is_uncertain && (
                                                <span className="text-[#ffb300]" title="Manual review recommended">⚠</span>
                                            )}
                                        </td>
                                    </motion.tr>
                                );
                            })}
                        </AnimatePresence>
                    </tbody>
                </table>

                {/* Empty state */}
                {batchResults.length === 0 && !isLoading && (
                    <div className="flex items-center justify-center h-64 text-[var(--color-text-muted)] text-sm font-mono uppercase tracking-widest">
                        Execute Batch Triage to populate
                    </div>
                )}

                {/* Loading state */}
                {isLoading && (
                    <div className="flex items-center justify-center h-64 gap-3">
                        <Loader2 className="animate-spin text-[#ff2a2a]" size={24} />
                        <span className="text-sm font-mono text-[var(--color-text-muted)] uppercase tracking-widest">
                            Running Rust LightGBM FFI...
                        </span>
                    </div>
                )}
            </div>
        </div>
    );
}
