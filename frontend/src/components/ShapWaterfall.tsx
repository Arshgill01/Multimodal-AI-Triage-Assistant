import { motion, AnimatePresence } from "framer-motion";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, ReferenceLine } from "recharts";
import { useAppStore } from "@/lib/store";

interface CustomTooltipProps {
    active?: boolean;
    payload?: any[];
    label?: string;
}

const CustomTooltip = ({ active, payload, label }: CustomTooltipProps) => {
    if (active && payload && payload.length) {
        const data = payload[0].payload;
        const isPositive = data.shap_value > 0;
        return (
            <div className="glass-panel p-3 text-sm">
                <p className="font-mono text-[var(--color-text-secondary)] mb-1 uppercase tracking-wider">{label}</p>
                <p className="flex justify-between gap-4">
                    <span className="text-[var(--color-text-muted)]">Value:</span>
                    <span className="font-mono text-[var(--color-text-primary)]">{data.value.toFixed(2)}</span>
                </p>
                <p className="flex justify-between gap-4">
                    <span className="text-[var(--color-text-muted)]">Impact:</span>
                    <span className={`font-mono font-bold ${isPositive ? 'text-[#ff2a2a]' : 'text-[#00e5ff]'}`}>
                        {isPositive ? '+' : ''}{data.shap_value.toFixed(3)}
                    </span>
                </p>
            </div>
        );
    }
    return null;
};

export default function ShapWaterfall() {
    const prediction = useAppStore((state) => state.prediction);

    if (!prediction || !prediction.shap) return null;

    const features = [...prediction.shap.features].sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value));

    // Cap at top 7 features for visual clarity
    const topFeatures = features.slice(0, 7).reverse();

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5, duration: 0.8 }}
            className="w-full h-full min-h-[300px]"
        >
            <h3 className="text-xs uppercase tracking-[0.2em] text-[var(--color-text-secondary)] mb-4 flex justify-between">
                <span>Feature Attribution (SHAP)</span>
                <span className="text-[var(--color-text-muted)]">Base: {prediction.shap.base_value.toFixed(2)}</span>
            </h3>

            <ResponsiveContainer width="100%" height="85%">
                <BarChart
                    data={topFeatures}
                    layout="vertical"
                    margin={{ top: 5, right: 30, left: 40, bottom: 5 }}
                >
                    <XAxis type="number" hide domain={['dataMin - 0.5', 'dataMax + 0.5']} />
                    <YAxis
                        dataKey="name"
                        type="category"
                        axisLine={false}
                        tickLine={false}
                        tick={{ fill: 'var(--color-text-secondary)', fontSize: 11, fontFamily: 'monospace' }}
                        width={80}
                    />
                    <Tooltip cursor={{ fill: 'rgba(255,255,255,0.02)' }} content={<CustomTooltip />} />
                    <ReferenceLine x={0} stroke="var(--color-obsidian-border)" strokeDasharray="3 3" />
                    <Bar dataKey="shap_value" radius={[0, 4, 4, 0]} barSize={24} isAnimationActive={true} animationDuration={1500} animationEasing="ease-out">
                        {topFeatures.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.shap_value > 0 ? '#ff2a2a' : '#00e5ff'} className="opacity-80 hover:opacity-100 transition-opacity" />
                        ))}
                    </Bar>
                </BarChart>
            </ResponsiveContainer>
        </motion.div>
    );
}
