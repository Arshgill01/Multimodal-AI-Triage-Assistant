import { useAppStore } from "@/lib/store";
import VitalSlider from "./VitalSlider";
import { Terminal } from "lucide-react";

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
        <div className="flex flex-col h-full justify-between">
            {/* Vitals Section */}
            <div className="flex-1 overflow-y-auto pr-2 pb-4">
                <VitalSlider
                    label="Heart Rate"
                    value={patientData.heart_rate}
                    min={30}
                    max={200}
                    unit="bpm"
                    onChange={(v) => updatePatient("heart_rate", v)}
                    isAbnormal={patientData.heart_rate > 100 || patientData.heart_rate < 50}
                />
                <VitalSlider
                    label="Resp Rate"
                    value={patientData.resp_rate}
                    min={8}
                    max={50}
                    unit="rpm"
                    onChange={(v) => updatePatient("resp_rate", v)}
                    isAbnormal={patientData.resp_rate > 24 || patientData.resp_rate < 12}
                />
                <VitalSlider
                    label="SpO2"
                    value={patientData.spo2}
                    min={70}
                    max={100}
                    unit="%"
                    onChange={(v) => updatePatient("spo2", v)}
                    isAbnormal={patientData.spo2 < 92}
                />
                <VitalSlider
                    label="Temp"
                    value={patientData.temp_f}
                    min={92}
                    max={106}
                    unit="°F"
                    onChange={(v) => updatePatient("temp_f", v)}
                    isAbnormal={patientData.temp_f > 100.4 || patientData.temp_f < 95}
                />
                <VitalSlider
                    label="Systolic BP"
                    value={patientData.systolic_bp}
                    min={60}
                    max={220}
                    unit="mmHg"
                    onChange={(v) => updatePatient("systolic_bp", v)}
                    isAbnormal={patientData.systolic_bp < 90 || patientData.systolic_bp > 160}
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
                    isAbnormal={patientData.pain_scale >= 7}
                />
            </div>

            {/* Terminal Chief Complaint Input */}
            <div className="mt-4 shrink-0">
                <label className="text-xs tracking-wider text-[var(--color-text-secondary)] uppercase mb-2 flex items-center gap-2">
                    <Terminal size={14} /> Sequence: Chief Complaint
                </label>
                <textarea
                    className="w-full bg-[rgba(0,0,0,0.3)] border border-[var(--color-obsidian-border)] rounded-lg p-3 text-sm font-mono text-[var(--color-text-primary)] focus:outline-none focus:border-[var(--color-text-secondary)] transition-colors h-24 resize-none"
                    value={patientData.chief_complaint}
                    onChange={(e) => updatePatient("chief_complaint", e.target.value)}
                    placeholder="> Enter patient reported symptoms..."
                />
            </div>

            {/* Primary Action Button */}
            <button
                onClick={handleEvaluate}
                disabled={analysisPhase !== "idle" && analysisPhase !== "complete"}
                className="mt-6 shrink-0 w-full py-4 glass-panel hover:bg-[rgba(255,255,255,0.08)] transition-all font-bold tracking-widest text-[#00e5ff] uppercase disabled:opacity-50 disabled:cursor-not-allowed group relative overflow-hidden"
            >
                <span className="relative z-10">Evaluate Patient</span>
                <div className="absolute inset-0 bg-gradient-to-r from-transparent via-[#00e5ff] to-transparent opacity-0 group-hover:opacity-10 transition-opacity transform -translate-x-full group-hover:animate-[shimmer_1.5s_infinite]" />
            </button>
        </div>
    );
}
