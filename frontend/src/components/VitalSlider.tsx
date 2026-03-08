import { motion } from "framer-motion";

interface VitalSliderProps {
    label: string;
    value: number;
    min: number;
    max: number;
    unit: string;
    onChange: (val: number) => void;
    // Optional flag to add a visual pulse effect if value is abnormal
    isAbnormal?: boolean;
}

export default function VitalSlider({
    label,
    value,
    min,
    max,
    unit,
    onChange,
    isAbnormal = false,
}: VitalSliderProps) {
    const percentage = ((value - min) / (max - min)) * 100;

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
                        className={`text-lg font-mono font-bold ${isAbnormal ? "text-[var(--color-esi-2)]" : "text-[var(--color-text-primary)]"
                            }`}
                    >
                        {value.toFixed(1).replace(/\.0$/, "")}
                    </motion.span>
                    <span className="text-[10px] text-[var(--color-text-muted)] uppercase tracking-widest">
                        {unit}
                    </span>
                </div>
            </div>

            <div className="relative h-1.5 w-full bg-[rgba(255,255,255,0.05)] rounded-full overflow-hidden">
                <motion.div
                    className="absolute top-0 left-0 h-full bg-[var(--color-text-secondary)]"
                    layoutId={`fill-${label}`}
                    initial={false}
                    animate={{ width: `${percentage}%` }}
                    transition={{ type: "spring", stiffness: 300, damping: 30 }}
                />
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
        </div>
    );
}
