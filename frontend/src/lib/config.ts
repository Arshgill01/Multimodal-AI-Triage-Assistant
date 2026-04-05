/**
 * Centralized API configuration for the Frostbyte Obsidian HUD.
 * All backend URLs are defined here to avoid hardcoded strings in components.
 */

export const RUST_API = process.env.NEXT_PUBLIC_RUST_API || "http://localhost:3001";
export const PYTHON_API = process.env.NEXT_PUBLIC_PYTHON_API || "http://localhost:8000";
