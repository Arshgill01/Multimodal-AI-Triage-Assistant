import { useState, useRef } from "react";
import { useAppStore } from "@/lib/store";
import { Image as ImageIcon, UploadCloud, X } from "lucide-react";

export default function ImageDropzone() {
    const { patientData, updatePatient } = useAppStore();
    const [isDragging, setIsDragging] = useState(false);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleFile = (file: File) => {
        if (!file.type.startsWith("image/")) return;

        const reader = new FileReader();
        reader.onloadend = () => {
            const base64String = reader.result as string;
            // Strip the data:image/jpeg;base64, prefix for the backend
            const base64Data = base64String.split(",")[1];
            updatePatient("image_base64", base64Data);
        };
        reader.readAsDataURL(file);
    };

    return (
        <div className="mt-6 shrink-0">
            <h3 className="text-[10px] tracking-widest uppercase text-[var(--color-text-secondary)] mb-2 flex items-center gap-2">
                <ImageIcon size={12} /> Clinical Imagery (Optional)
            </h3>

            {patientData.image_base64 ? (
                <div className="relative w-full h-24 rounded-lg overflow-hidden border border-[var(--color-obsidian-border)] group">
                    {/* Reconstruct data URI to show preview */}
                    <img 
                        src={`data:image/jpeg;base64,${patientData.image_base64}`} 
                        alt="Clinical upload" 
                        className="w-full h-full object-cover opacity-60" 
                    />
                    <div className="absolute inset-0 bg-black/40 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                        <button 
                            onClick={() => updatePatient("image_base64", undefined)}
                            className="bg-black/80 hover:bg-[#ff2a2a]/20 text-white p-2 rounded-full border border-white/10 transition-colors"
                        >
                            <X size={16} />
                        </button>
                    </div>
                </div>
            ) : (
                <div 
                    className={`w-full h-24 rounded-lg border-2 border-dashed transition-all flex flex-col items-center justify-center cursor-pointer ${
                        isDragging 
                            ? 'border-[#00e5ff] bg-[#00e5ff]/5' 
                            : 'border-[var(--color-obsidian-border)] bg-[rgba(0,0,0,0.2)] hover:border-[var(--color-text-secondary)]'
                    }`}
                    onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                    onDragLeave={() => setIsDragging(false)}
                    onDrop={(e) => {
                        e.preventDefault();
                        setIsDragging(false);
                        const file = e.dataTransfer.files[0];
                        if (file) handleFile(file);
                    }}
                    onClick={() => fileInputRef.current?.click()}
                >
                    <UploadCloud size={20} className={`mb-2 ${isDragging ? 'text-[#00e5ff]' : 'text-[var(--color-text-muted)]'}`} />
                    <p className="text-xs text-[var(--color-text-muted)] tracking-wider">Drag Drop or Click</p>
                    <input 
                        type="file" 
                        className="hidden" 
                        accept="image/*" 
                        ref={fileInputRef}
                        onChange={(e) => {
                            const file = e.target.files?.[0];
                            if (file) handleFile(file);
                        }}
                    />
                </div>
            )}
        </div>
    );
}
