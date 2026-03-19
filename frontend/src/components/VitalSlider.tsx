import { motion } from "framer-motion";
import { useMemo } from "react";

/**
 * Clinical range zones for a single vital sign.
 * Each zone maps an absolute vital value to a severity color.
 * Zones MUST be ordered by ascending threshold.
 */
interface ClinicalZone {
    /** Upper bound (exclusive) for this zone */
    upTo: number;
    /** CSS color string */
    color: string;
}

interface VitalSliderProps {
    label: string;
    value: number;
    min: number;
    max: number;
    unit: string;
    onChange: (val: number) => void;
    /**
     * Clinical range zones — ordered array of { upTo, color }.
     * The last zone implicitly covers everything above its predecessor.
     * Example for Heart Rate:
     *   [{ upTo: 50, color: "#ff2a2a" }, { upTo: 60, color: "#ffb300" },
     *    { upTo: 100, color: "#00c853" }, { upTo: 120, color: "#ffb300" },
     *    { upTo: 200, color: "#ff2a2a" }]
     */
    zones?: ClinicalZone[];
    /** Legacy fallback: simple abnormal flag */
    isAbnormal?: boolean;
}

export default function VitalSlider({
    label,
    value,
    min,
    max,
    unit,
    onChange,
    zones,
    isAbnormal = false,
}: VitalSliderProps) {
    const percentage = ((value - min) / (max - min)) * 100;

    // Build the CSS linear-gradient string from clinical zones
    const trackGradient = useMemo(() => {
        if (!zones || zones.length === 0) return null;
        const range = max - min;
        const stops: string[] = [];
        for (const zone of zones) {
            const pct = ((zone.upTo - min) / range) * 100;
            stops.push(`${zone.color} ${Math.max(0, Math.min(100, pct))}%`);
        }
        return `linear-gradient(to right, ${stops.join(", ")})`;
    }, [zones, min, max]);

    // Determine the current zone color for the value display
    const currentColor = useMemo(() => {
        if (!zones || zones.length === 0) {
            return isAbnormal ? "#ff9100" : "var(--color-text-primary)";
        }
        for (const zone of zones) {
            if (value < zone.upTo) return zone.color;
        }
        return zones[zones.length - 1].color;
    }, [zones, value, isAbnormal]);

    // Determine if the value is in a critical (red) zone for pulsing
    const isCritical = useMemo(() => {
        if (!zones || zones.length === 0) return false;
        for (const zone of zones) {
            if (value < zone.upTo) return zone.color === "#ff2a2a";
        }
        return zones[zones.length - 1].color === "#ff2a2a";
    }, [zones, value]);

    return (
        <div className="flex flex-col mb-5">
            <div className="flex justify-between items-end mb-2">
                <label className="text-xs tracking-wider text-[var(--color-text-secondary)] uppercase">
                    {label}
                </label>
                <div className="flex items-baseline gap-1">
                    <motion.span
                        key={value}
                        initial={{ y: -5, opacity: 0 }}
                        animate={{ y: 0, opacity: 1 }}
                        className="text-lg font-mono font-bold"
                        style={{
                            color: currentColor,
                            textShadow: isCritical ? `0 0 12px ${currentColor}` : "none",
                        }}
                    >
                        {value.toFixed(1).replace(/\.0$/, "")}
                    </motion.span>
                    <span className="text-[10px] text-[var(--color-text-muted)] uppercase tracking-widest">
                        {unit}
                    </span>
                </div>
            </div>

            <div className="relative h-2 w-full rounded-full overflow-hidden">
                {/* Background: either clinical gradient or plain dark */}
                <div
                    className="absolute inset-0 rounded-full"
                    style={{
                        background: trackGradient || "rgba(255,255,255,0.05)",
                        opacity: trackGradient ? 0.25 : 1,
                    }}
                />

                {/* Active fill bar */}
                <motion.div
                    className="absolute top-0 left-0 h-full rounded-full"
                    layoutId={`fill-${label}`}
                    initial={false}
                    animate={{ width: `${percentage}%` }}
                    transition={{ type: "spring", stiffness: 300, damping: 30 }}
                    style={{
                        background: trackGradient || "var(--color-text-secondary)",
                    }}
                />

                {/* Invisible native range input on top */}
                <input
                    type="range"
                    min={min}
                    max={max}
                    step={label.includes("Temp") ? 0.1 : 1}
                    value={value}
                    onChange={(e) => onChange(parseFloat(e.target.value))}
                    className="absolute inset-0 w-full opacity-0 cursor-ew-resize"
                />
            </div>

            {/* Critical zone pulse indicator */}
            {isCritical && (
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: [0.4, 1, 0.4] }}
                    transition={{ duration: 1.2, repeat: Infinity }}
                    className="mt-1.5 text-[9px] font-mono uppercase tracking-[0.3em] text-[#ff2a2a]"
                >
                    ⚠ Critical Range
                </motion.div>
            )}
        </div>
    );
}
