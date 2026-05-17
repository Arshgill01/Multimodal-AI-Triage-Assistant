"use client";

import { useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useAppStore } from "@/lib/store";
import type { AuditEntry } from "@/lib/api-types";
import { 
    ShieldCheck, AlertTriangle, Activity, RefreshCw, 
    Filter, CheckCircle, XCircle, Clock 
} from "lucide-react";

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

function formatTimestamp(ts: string) {
    const d = new Date(ts);
    return d.toLocaleString("en-US", { 
        month: "short", 
        day: "numeric", 
        hour: "2-digit", 
        minute: "2-digit" 
    });
}

function AuditRow({ entry }: { entry: AuditEntry }) {
    return (
        <motion.tr
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="border-b border-[rgba(255,255,255,0.05)] hover:bg-[rgba(255,255,255,0.02)] transition-colors"
        >
            <td className="px-3 py-3 text-xs text-[var(--color-text-muted)] font-mono">
                {formatTimestamp(entry.timestamp)}
            </td>
            <td className="px-3 py-3">
                <span
                    className="inline-flex items-center justify-center w-7 h-7 rounded font-bold text-sm"
                    style={{ 
                        color: ESI_COLORS[entry.predicted_esi], 
                        border: `1px solid ${ESI_COLORS[entry.predicted_esi]}`,
                        backgroundColor: `${ESI_COLORS[entry.predicted_esi]}15`
                    }}
                >
                    {entry.predicted_esi}
                </span>
            </td>
            <td className="px-3 py-3 text-xs text-[var(--color-text-primary)] max-w-[200px] truncate">
                {entry.chief_complaint || "—"}
            </td>
            <td className="px-3 py-3 text-right">
                <span className={`text-xs font-mono ${
                    entry.is_uncertain 
                        ? "text-[#ffb300]" 
                        : entry.confidence > 0.8 
                            ? "text-[#00c853]" 
                            : "text-[var(--color-text-secondary)]"
                }`}>
                    {(entry.confidence * 100).toFixed(1)}%
                </span>
                {entry.is_uncertain && (
                    <span className="ml-1 text-[#ffb300]" title="Manual review recommended">⚠</span>
                )}
            </td>
            <td className="px-3 py-3">
                <div className="flex flex-wrap gap-1 max-w-[180px]">
                    {entry.top_shap_drivers.slice(0, 3).map((driver, i) => (
                        <span 
                            key={i}
                            className="text-[10px] px-1.5 py-0.5 bg-[rgba(0,229,255,0.1)] text-[#00e5ff] rounded font-mono"
                        >
                            {driver}
                        </span>
                    ))}
                    {entry.top_shap_drivers.length > 3 && (
                        <span className="text-[10px] text-[var(--color-text-muted)]">
                            +{entry.top_shap_drivers.length - 3}
                        </span>
                    )}
                </div>
            </td>
            <td className="px-3 py-3 text-center">
                {entry.overridden ? (
                    <div className="flex flex-col items-center gap-1">
                        <CheckCircle size={14} className="text-[#00c853]" />
                        <span className="text-[10px] font-mono text-[#00c853]">
                            ESI {entry.override_esi}
                        </span>
                    </div>
                ) : (
                    <XCircle size={14} className="text-[var(--color-text-muted)] opacity_30" />
                )}
            </td>
            <td className="px-3 py-3 max-w-[150px]">
                {entry.override_reason ? (
                    <span className="text-[10px] text-[var(--color-text-muted)] italic">
                        {entry.override_reason}
                    </span>
                ) : (
                    <span className="text-[10px] text-[var(--color-text-muted)] opacity_30">—</span>
                )}
            </td>
        </motion.tr>
    );
}

export default function TrustConsole() {
    const { 
        auditEntries, auditSummary, auditLoading, auditFilter,
        setAuditFilter, fetchAuditLog, fetchAuditSummary 
    } = useAppStore();

    useEffect(() => {
        fetchAuditLog();
        fetchAuditSummary();
    }, []);
    useEffect(() => {
   fetchAuditLog();
}, [auditFilter]);

    const handleFilterChange = (key: keyof typeof auditFilter, value: any) => {
        setAuditFilter({ ...auditFilter, [key]: value });
       
    };

    return (
        <div className="w-full h-full flex flex-col">
            {/* Header */}
            <div className="flex items-center justify-between mb-6 pb-4 border-b border-[var(--color-obsidian-border)]">
                <div className="flex items-center gap-3">
                    <ShieldCheck className="text-[#00e5ff]" size={20} />
                    <h1 className="text-lg font-bold tracking-widest uppercase text-[#00e5ff]">
                        Trust Console
                    </h1>
                    <span className="text-xs text-[var(--color-text-muted)] font-mono">
                        Audit Review Surface
                    </span>
                </div>

                <div className="flex items-center gap-3">
                    <button
                        onClick={() => { fetchAuditLog(); fetchAuditSummary(); }}
                        className="flex items-center gap-1.5 px-3 py-2 text-[10px] font-mono uppercase tracking-widest text-[var(--color-text-muted)] hover:text-[var(--color-text-secondary)] transition-colors border border-[var(--color-obsidian-border)] rounded-lg bg-[rgba(255,255,255,0.02)] hover:bg-[rgba(255,255,255,0.05)]"
                    >
                        <RefreshCw size={10} /> Refresh
                    </button>
                </div>
            </div>

            {/* Trust Metrics Cards */}
            <div className="grid grid-cols-4 gap-4 mb-6">
                <motion.div 
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="glass-panel rounded-lg p-4 border border-[var(--color-obsidian-border)]"
                >
                    <div className="flex items-center gap-2 mb-2">
                        <Activity size={12} className="text-[#00e5ff]" />
                        <span className="text-[10px] uppercase tracking-widest text-[var(--color-text-muted)]">
                            Total Cases
                        </span>
                    </div>
                    <div className="text-2xl font-bold text-[var(--color-text-primary)] font-mono">
                        {auditSummary?.total_cases ?? "—"}
                    </div>
                </motion.div>

                <motion.div 
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.1 }}
                    className="glass-panel rounded-lg p-4 border border-[var(--color-obsidian-border)]"
                >
                    <div className="flex items-center gap-2 mb-2">
                        <AlertTriangle size={12} className="text-[#ffb300]" />
                        <span className="text-[10px] uppercase tracking-widest text-[var(--color-text-muted)]">
                            Uncertain Cases
                        </span>
                    </div>
                    <div className="text-2xl font-bold text-[#ffb300] font-mono">
                        {auditSummary?.uncertain_count ?? "—"}
                    </div>
                </motion.div>

                <motion.div 
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.2 }}
                    className="glass-panel rounded-lg p-4 border border-[var(--color-obsidian-border)]"
                >
                    <div className="flex items-center gap-2 mb-2">
                        <CheckCircle size={12} className="text-[#00c853]" />
                        <span className="text-[10px] uppercase tracking-widest text-[var(--color-text-muted)]">
                            Clinician Overrides
                        </span>
                    </div>
                    <div className="text-2xl font-bold text-[#00c853] font-mono">
                        {auditSummary?.override_count ?? "—"}
                    </div>
                </motion.div>

                <motion.div 
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.3 }}
                    className="glass-panel rounded-lg p-4 border border-[var(--color-obsidian-border)]"
                >
                    <div className="flex items-center gap-2 mb-2">
                        <Clock size={12} className="text-[var(--color-text-muted)]" />
                        <span className="text-[10px] uppercase tracking-widest text-[var(--color-text-muted)]">
                            Override Rate
                        </span>
                    </div>
                    <div className="text-2xl font-bold text-[var(--color-text-primary)] font-mono">
                        {auditSummary && auditSummary.total_cases > 0 
                            ? ((auditSummary.override_count / auditSummary.total_cases) * 100).toFixed(1)
                            : "—"
                        }%
                    </div>
                </motion.div>
            </div>

            {/* Filters */}
            <div className="flex items-center gap-4 mb-4 pb-4 border-b border-[var(--color-obsidian-border)]">
                <div className="flex items-center gap-2">
                    <Filter size={12} className="text-[var(--color-text-muted)]" />
                    <span className="text-[10px] uppercase tracking-widest text-[var(--color-text-muted)]">
                        Filters:
                    </span>
                </div>

                <div className="flex gap-2">
                    <button
                        onClick={() => handleFilterChange("esi_filter", undefined)}
                        className={`px-3 py-1.5 text-[10px] font-mono uppercase tracking-widest rounded border transition-colors ${
                            !auditFilter.esi_filter
                                ? "bg-[rgba(0,229,255,0.1)] text-[#00e5ff] border-[#00e5ff]"
                                : "bg-[rgba(255,255,255,0.02)] text-[var(--color-text-muted)] border-[var(--color-obsidian-border)] hover:border-[var(--color-text-secondary)]"
                        }`}
                    >
                        All ESI
                    </button>
                    {[1, 2, 3, 4, 5].map((esi) => (
                        <button
                            key={esi}
                            onClick={() => handleFilterChange("esi_filter", esi)}
                            className={`px-3 py-1.5 text-[10px] font-mono uppercase tracking-widest rounded border transition-colors ${
                                auditFilter.esi_filter === esi
                                    ? "bg-[rgba(0,229,255,0.1)] text-[#00e5ff] border-[#00e5ff]"
                                    : "bg-[rgba(255,255,255,0.02)] text-[var(--color-text-muted)] border-[var(--color-obsidian-border)] hover:border-[var(--color-text-secondary)]"
                            }`}
                            style={auditFilter.esi_filter === esi ? { color: ESI_COLORS[esi], borderColor: ESI_COLORS[esi] } : {}}
                        >
                            ESI {esi}
                        </button>
                    ))}
                </div>

                <div className="h-4 w-px bg-[var(--color-obsidian-border)]" />

                <button
                    onClick={() => handleFilterChange("uncertain_only", !auditFilter.uncertain_only)}
                    className={`px-3 py-1.5 text-[10px] font-mono uppercase tracking-widest rounded border transition-colors ${
                        auditFilter.uncertain_only
                            ? "bg-[rgba(255,179,0,0.1)] text-[#ffb300] border-[#ffb300]"
                            : "bg-[rgba(255,255,255,0.02)] text-[var(--color-text-muted)] border-[var(--color-obsidian-border)] hover:border-[var(--color-text-secondary)]"
                    }`}
                >
                    Uncertain Only
                </button>

                <button
                    onClick={() => handleFilterChange("overridden_only", !auditFilter.overridden_only)}
                    className={`px-3 py-1.5 text-[10px] font-mono uppercase tracking-widest rounded border transition-colors ${
                        auditFilter.overridden_only
                            ? "bg-[rgba(0,200,83,0.1)] text-[#00c853] border-[#00c853]"
                            : "bg-[rgba(255,255,255,0.02)] text-[var(--color-text-muted)] border-[var(--color-obsidian-border)] hover:border-[var(--color-text-secondary)]"
                    }`}
                >
                    Overridden Only
                </button>
            </div>

            {/* Audit Log Table */}
            <div className="flex-1 overflow-auto rounded-lg border border-[var(--color-obsidian-border)]">
                <table className="w-full text-xs">
                    <thead className="sticky top-0 bg-[rgba(7,7,9,0.95)] border-b border-[var(--color-obsidian-border)]">
                        <tr className="text-[var(--color-text-muted)] uppercase tracking-widest text-[10px]">
                            <th className="text-left px-3 py-2 font-medium">Timestamp</th>
                            <th className="text-left px-3 py-2 font-medium">ESI</th>
                            <th className="text-left px-3 py-2 font-medium">Chief Complaint</th>
                            <th className="text-right px-3 py-2 font-medium">Confidence</th>
                            <th className="text-left px-3 py-2 font-medium">Top SHAP Drivers</th>
                            <th className="text-center px-3 py-2 font-medium">Override</th>
                            <th className="text-left px-3 py-2 font-medium">Reason</th>
                        </tr>
                    </thead>
                    <tbody>
                        <AnimatePresence>
                            {auditLoading ? (
                                <tr>
                                    <td colSpan={7} className="text-center py-8">
                                        <div className="flex items-center justify-center gap-2 text-[var(--color-text-muted)]">
                                            <RefreshCw size={14} className="animate-spin" />
                                            <span className="font-mono text-xs uppercase">Loading audit log...</span>
                                        </div>
                                    </td>
                                </tr>
                            ) : auditEntries.length === 0 ? (
                                <tr>
                                    <td colSpan={7} className="text-center py-8">
                                        <div className="text-[var(--color-text-muted)] font-mono text-xs uppercase">
                                            No audit entries found
                                        </div>
                                    </td>
                                </tr>
                            ) : (
                                auditEntries.map((entry) => (
                                    <AuditRow key={entry.id} entry={entry} />
                                ))
                            )}
                        </AnimatePresence>
                    </tbody>
                </table>
            </div>

            {/* Footer */}
            <div className="mt-4 pt-4 border-t border-[var(--color-obsidian-border)] flex items-center justify-between">
                <div className="text-[10px] text-[var(--color-text-muted)] font-mono">
                    Showing {auditEntries.length} recent triage decisions
                </div>
                <div className="text-[10px] text-[var(--color-text-muted)] font-mono">
                    Data refreshes on load • Filter persists during session
                </div>
            </div>
        </div>
    );
}