import { create } from "zustand";
import { RUST_API } from "@/lib/config";
import type { PredictResponse, BatchPatientResult, SimilarCaseEvidence, AuditEntry, AuditSummary } from "@/lib/api-types";

export type AnalysisPhase = "idle" | "extracting" | "routing" | "inferring" | "explainability" | "rag" | "complete";
export type EsiLevel = 1 | 2 | 3 | 4 | 5 | null;

export type AuditFilter = {
    esi_filter?: number;
    uncertain_only?: boolean;
    overridden_only?: boolean;
};

interface PatientState {
    age: number;
    heart_rate: number;
    resp_rate: number;
    spo2: number;
    temp_f: number;
    systolic_bp: number;
    pain_scale: number;
    chief_complaint: string;
    image_base64?: string;
}

interface AppState {
    // UI State
    currentEsi: EsiLevel;
    analysisPhase: AnalysisPhase;
    isMciMode: boolean;
    isTrustConsoleMode: boolean;

    // Inputs
    patientData: PatientState;

    // Results
    prediction: PredictResponse | null;
    ragStream: string;
    similarCases: SimilarCaseEvidence[];
    batchResults: BatchPatientResult[];

    // Audit / Trust Console State
    auditEntries: AuditEntry[];
    auditSummary: AuditSummary | null;
    auditLoading: boolean;
    auditFilter: AuditFilter;

    // Actions
    setEsi: (esi: EsiLevel) => void;
    setPhase: (phase: AnalysisPhase) => void;
    updatePatient: (field: keyof PatientState, value: any) => void;
    setPrediction: (pred: PredictResponse | null) => void;
    appendRagStream: (chunk: string) => void;
    setSimilarCases: (cases: SimilarCaseEvidence[]) => void;
    setMciMode: (on: boolean) => void;
    setTrustConsoleMode: (on: boolean) => void;
    setBatchResults: (results: BatchPatientResult[]) => void;
    setAuditFilter: (filter: AuditFilter) => void;
    fetchAuditLog: () => Promise<void>;
    fetchAuditSummary: () => Promise<void>;
    reset: () => void;
}

const defaultPatient: PatientState = {
    age: 45,
    heart_rate: 80,
    resp_rate: 16,
    spo2: 98,
    temp_f: 98.6,
    systolic_bp: 120,
    pain_scale: 0,
    chief_complaint: "",
    image_base64: undefined,
};

export const useAppStore = create<AppState>((set, get) => ({
    currentEsi: null,
    analysisPhase: "idle",
    isMciMode: false,
    isTrustConsoleMode: false,

    patientData: { ...defaultPatient },

    prediction: null,
    ragStream: "",
    similarCases: [],
    batchResults: [],

    auditEntries: [],
    auditSummary: null,
    auditLoading: false,
    auditFilter: {},

    setEsi: (esi) => set({ currentEsi: esi }),
    setPhase: (phase) => set({ analysisPhase: phase }),
    updatePatient: (field, value) =>
        set((state) => ({ patientData: { ...state.patientData, [field]: value } })),
    setPrediction: (pred) => set({ prediction: pred }),
    appendRagStream: (chunk) => set((state) => ({ ragStream: state.ragStream + chunk })),
    setSimilarCases: (cases) => set({ similarCases: cases }),
    setMciMode: (on) => set({ isMciMode: on, isTrustConsoleMode: false }),
    setTrustConsoleMode: (on) => set({ isTrustConsoleMode: on, isMciMode: false }),
    setBatchResults: (results) => set({ batchResults: results }),
    setAuditFilter: (filter) => set({ auditFilter: filter }),
    fetchAuditLog: async () => {
        set({ auditLoading: true });
        try {
            const { auditFilter } = get();
            const params = new URLSearchParams();
            params.append("limit", "50");
            if (auditFilter.esi_filter) params.append("esi_filter", auditFilter.esi_filter.toString());
            if (auditFilter.uncertain_only) params.append("uncertain_only", "true");
            if (auditFilter.overridden_only) params.append("overridden_only", "true");
            
            const resp = await fetch(`${RUST_API}/audit-log?${params.toString()}`);
            const data = await resp.json();
            set({ auditEntries: data.entries, auditLoading: false });
        } catch (err) {
            console.error("Failed to fetch audit log:", err);
            set({ auditLoading: false });
        }
    },
    fetchAuditSummary: async () => {
        try {
            const resp = await fetch(`${RUST_API}/audit-summary`);
            const data = await resp.json();
            set({ auditSummary: data });
        } catch (err) {
            console.error("Failed to fetch audit summary:", err);
        }
    },

    reset: () => set({
        currentEsi: null,
        analysisPhase: "idle",
        isMciMode: false,
        isTrustConsoleMode: false,
        patientData: { ...defaultPatient },
        prediction: null,
        ragStream: "",
        similarCases: [],
        batchResults: [],
    }),
}));
