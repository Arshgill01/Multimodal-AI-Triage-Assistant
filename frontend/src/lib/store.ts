import { create } from "zustand";

export type AnalysisPhase = "idle" | "extracting" | "routing" | "inferring" | "explainability" | "rag" | "complete";
export type EsiLevel = 1 | 2 | 3 | 4 | 5 | null;

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

    // Inputs
    patientData: PatientState;

    // Results
    prediction: any | null;
    ragStream: string;
    similarCases: any[];
    batchResults: any[];

    // Actions
    setEsi: (esi: EsiLevel) => void;
    setPhase: (phase: AnalysisPhase) => void;
    updatePatient: (field: keyof PatientState, value: any) => void;
    setPrediction: (pred: any) => void;
    appendRagStream: (chunk: string) => void;
    setSimilarCases: (cases: any[]) => void;
    setMciMode: (on: boolean) => void;
    setBatchResults: (results: any[]) => void;
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

export const useAppStore = create<AppState>((set) => ({
    currentEsi: null,
    analysisPhase: "idle",
    isMciMode: false,

    patientData: { ...defaultPatient },

    prediction: null,
    ragStream: "",
    similarCases: [],
    batchResults: [],

    setEsi: (esi) => set({ currentEsi: esi }),
    setPhase: (phase) => set({ analysisPhase: phase }),
    updatePatient: (field, value) =>
        set((state) => ({ patientData: { ...state.patientData, [field]: value } })),
    setPrediction: (pred) => set({ prediction: pred }),
    appendRagStream: (chunk) => set((state) => ({ ragStream: state.ragStream + chunk })),
    setSimilarCases: (cases) => set({ similarCases: cases }),
    setMciMode: (on) => set({ isMciMode: on }),
    setBatchResults: (results) => set({ batchResults: results }),

    reset: () => set({
        currentEsi: null,
        analysisPhase: "idle",
        isMciMode: false,
        patientData: { ...defaultPatient },
        prediction: null,
        ragStream: "",
        similarCases: [],
        batchResults: [],
    }),
}));
