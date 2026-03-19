import { useAppStore } from "@/lib/store";
import VitalSlider from "./VitalSlider";
import ImageDropzone from "./ImageDropzone";
import { Terminal } from "lucide-react";

// ── Clinical Range Definitions ──────────────────────────────
// Based on standard ED triage thresholds (AHA/ACLS guidelines)
// Color key: #00c853 = safe, #ffb300 = caution, #ff2a2a = critical

const HEART_RATE_ZONES = [
    { upTo: 40, color: "#ff2a2a" },   // Severe bradycardia
    { upTo: 50, color: "#ffb300" },   // Bradycardia
    { upTo: 60, color: "#ffb300" },   // Low normal
    { upTo: 100, color: "#00c853" },  // Normal sinus
    { upTo: 120, color: "#ffb300" },  // Tachycardia
    { upTo: 200, color: "#ff2a2a" },  // SVT / unstable
];

const RESP_RATE_ZONES = [
    { upTo: 10, color: "#ff2a2a" },   // Hypoventilation
    { upTo: 12, color: "#ffb300" },
    { upTo: 20, color: "#00c853" },   // Normal
    { upTo: 24, color: "#ffb300" },
    { upTo: 50, color: "#ff2a2a" },   // Tachypnea
];

const SPO2_ZONES = [
    { upTo: 85, color: "#ff2a2a" },   // Severe hypoxia
    { upTo: 90, color: "#ff2a2a" },
    { upTo: 92, color: "#ffb300" },
    { upTo: 100, color: "#00c853" },  // Normal
];

const TEMP_ZONES = [
    { upTo: 95, color: "#ff2a2a" },   // Hypothermia
    { upTo: 96.8, color: "#ffb300" },
    { upTo: 99.5, color: "#00c853" }, // Normal
    { upTo: 100.4, color: "#ffb300" },// Low-grade fever
    { upTo: 103, color: "#ff2a2a" },  // High fever
    { upTo: 106, color: "#ff2a2a" },
];

const BP_ZONES = [
    { upTo: 80, color: "#ff2a2a" },   // Hypotensive shock
    { upTo: 90, color: "#ffb300" },   // Hypotension
    { upTo: 120, color: "#00c853" },  // Normal
    { upTo: 140, color: "#ffb300" },  // Prehypertension
    { upTo: 160, color: "#ff2a2a" },  // Stage 2 HTN
    { upTo: 220, color: "#ff2a2a" },  // Hypertensive crisis
];

const PAIN_ZONES = [
    { upTo: 1, color: "#00c853" },    // No pain
    { upTo: 4, color: "#00c853" },    // Mild
    { upTo: 7, color: "#ffb300" },    // Moderate
    { upTo: 10, color: "#ff2a2a" },   // Severe
];

export default function TelemetryPane() {
    const { patientData, updatePatient, setPhase, analysisPhase, setPrediction, setEsi } = useAppStore();

    const handleEvaluate = async () => {
        try {
            // 1. Initiate & show embeddings phase
            setPhase("extracting");

            // Simulate real-world embedding latency for UX comprehension
            await new Promise(r => setTimeout(r, 600));

            // 2. Routing phase
            setPhase("routing");
            await new Promise(r => setTimeout(r, 400));

            // 3. Inference phase
            setPhase("inferring");

            const response = await fetch("http://localhost:3001/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(patientData),
            });

            if (!response.ok) throw new Error("Inference failed");

            const data = await response.json();

            // 4. SHAP Explanation phase
            setPhase("explainability");
            await new Promise(r => setTimeout(r, 500));

            setPrediction(data);
            setEsi(data.predicted_esi as any);

            // 5. RAG phase (Wait for SSE to catch up)
            setPhase("rag");

            // Reset to complete shortly after RAG initializes
            setTimeout(() => setPhase("complete"), 2000);

        } catch (err) {
            console.error(err);
            setPhase("idle");
        }
    };

    return (
        <div className="flex flex-col h-full min-h-0">
            {/* Scrollable content area */}
            <div className="flex-1 min-h-0 overflow-y-auto pr-2 pb-4">
                <VitalSlider
                    label="Heart Rate"
                    value={patientData.heart_rate}
                    min={30}
                    max={200}
                    unit="bpm"
                    onChange={(v) => updatePatient("heart_rate", v)}
                    zones={HEART_RATE_ZONES}
                />
                <VitalSlider
                    label="Resp Rate"
                    value={patientData.resp_rate}
                    min={8}
                    max={50}
                    unit="rpm"
                    onChange={(v) => updatePatient("resp_rate", v)}
                    zones={RESP_RATE_ZONES}
                />
                <VitalSlider
                    label="SpO2"
                    value={patientData.spo2}
                    min={70}
                    max={100}
                    unit="%"
                    onChange={(v) => updatePatient("spo2", v)}
                    zones={SPO2_ZONES}
                />
                <VitalSlider
                    label="Temp"
                    value={patientData.temp_f}
                    min={92}
                    max={106}
                    unit="°F"
                    onChange={(v) => updatePatient("temp_f", v)}
                    zones={TEMP_ZONES}
                />
                <VitalSlider
                    label="Systolic BP"
                    value={patientData.systolic_bp}
                    min={60}
                    max={220}
                    unit="mmHg"
                    onChange={(v) => updatePatient("systolic_bp", v)}
                    zones={BP_ZONES}
                />
                <VitalSlider
                    label="Age"
                    value={patientData.age}
                    min={1}
                    max={105}
                    unit="yrs"
                    onChange={(v) => updatePatient("age", v)}
                />
                <VitalSlider
                    label="Pain Scale"
                    value={patientData.pain_scale}
                    min={0}
                    max={10}
                    unit="/10"
                    onChange={(v) => updatePatient("pain_scale", v)}
                    zones={PAIN_ZONES}
                />

                {/* Clinical Image Upload */}
                <ImageDropzone />

                {/* Chief Complaint — inside scrollable area */}
                <div className="mt-4 pt-4 border-t border-[rgba(255,255,255,0.05)]">
                    <label className="text-xs tracking-wider text-[var(--color-text-secondary)] uppercase mb-2 flex items-center gap-2">
                        <Terminal size={14} /> Sequence: Chief Complaint
                    </label>
                    <textarea
                        className="w-full bg-[rgba(0,0,0,0.3)] border border-[var(--color-obsidian-border)] rounded-lg p-3 text-sm font-mono text-[var(--color-text-primary)] focus:outline-none focus:border-[var(--color-text-secondary)] transition-colors h-20 resize-none"
                        value={patientData.chief_complaint}
                        onChange={(e) => updatePatient("chief_complaint", e.target.value)}
                        placeholder="> Enter patient reported symptoms..."
                    />
                </div>
            </div>

            {/* Primary Action Button — ONLY this stays pinned at bottom */}
            <button
                onClick={handleEvaluate}
                disabled={analysisPhase !== "idle" && analysisPhase !== "complete"}
                className="mt-3 shrink-0 w-full py-3 glass-panel hover:bg-[rgba(255,255,255,0.08)] transition-all font-bold tracking-widest text-[#00e5ff] uppercase disabled:opacity-50 disabled:cursor-not-allowed group relative overflow-hidden"
            >
                <span className="relative z-10">Evaluate Patient</span>
                <div className="absolute inset-0 bg-gradient-to-r from-transparent via-[#00e5ff] to-transparent opacity-0 group-hover:opacity-10 transition-opacity transform -translate-x-full group-hover:animate-[shimmer_1.5s_infinite]" />
            </button>
        </div>
    );
}
