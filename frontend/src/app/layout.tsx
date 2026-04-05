import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { cn } from "@/lib/utils";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });

export const metadata: Metadata = {
  title: "Frostbyte — Multimodal AI Triage Command Center",
  description: "Real-time ESI triage powered by ClinicalBERT, ResNet-50 vision, LightGBM via Rust FFI, SHAP explainability, and Gemini RAG intelligence.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className={cn(inter.variable, "antialiased font-sans h-screen w-screen")}>
        {children}
      </body>
    </html>
  );
}
