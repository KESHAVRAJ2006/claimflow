import type { Config } from "tailwindcss";

/**
 * Design tokens. Colours are CSS variables (RGB channels, see app/globals.css) so light and dark themes share one
 * set of class names and opacity modifiers like `bg-accent/10` still work.
 *
 * The config enforces parts of the brief so they can't be broken by accident:
 * - boxShadow defines only `sm` (and `none`): cards get shadow-sm and nothing heavier exists.
 * - backgroundImage is empty: no gradient utilities exist.
 * - named text sizes (title/section/body/table/label) carry size, line height, weight and tracking together.
 */
const rgb = (name: string) => `rgb(var(--${name}) / <alpha-value>)`;

const config: Config = {
  darkMode: "class",
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}", "./lib/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        background: rgb("background"),
        surface: rgb("surface"),
        "surface-muted": rgb("surface-muted"),
        border: rgb("border"),
        foreground: rgb("foreground"),
        muted: rgb("muted"),
        subtle: rgb("subtle"),
        ring: rgb("ring"),
        // The single accent: the one primary action per view.
        accent: { DEFAULT: rgb("accent"), hover: rgb("accent-hover"), foreground: rgb("accent-foreground") },
        // Status colours are separate from the accent and never used for actions.
        approve: { DEFAULT: rgb("approve"), soft: rgb("approve-soft") },
        reject: { DEFAULT: rgb("reject"), soft: rgb("reject-soft") },
        escalate: { DEFAULT: rgb("escalate"), soft: rgb("escalate-soft") },
      },
      fontFamily: {
        sans: ["var(--font-inter)", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "ui-monospace", "SFMono-Regular", "monospace"],
      },
      fontSize: {
        title: ["1.5rem", { lineHeight: "2rem", letterSpacing: "-0.02em", fontWeight: "600" }],
        section: ["1rem", { lineHeight: "1.5rem", fontWeight: "600" }],
        body: ["0.875rem", { lineHeight: "1.6" }],
        table: ["0.8125rem", { lineHeight: "1.25rem" }],
        label: ["0.75rem", { lineHeight: "1rem", letterSpacing: "0.05em", fontWeight: "500" }],
      },
      spacing: { sidebar: "220px", rail: "64px" },
      maxWidth: { content: "1280px" },
      borderRadius: { card: "8px" },
      transitionTimingFunction: { "out-expo": "cubic-bezier(0.22, 1, 0.36, 1)" },
      keyframes: {
        "pulse-line": { "0%, 100%": { opacity: "1" }, "50%": { opacity: "0.35" } },
      },
      animation: { "pulse-line": "pulse-line 1.6s cubic-bezier(0.4, 0, 0.6, 1) infinite" },
    },
    // Replaced (not extended): only a subtle shadow exists.
    boxShadow: { none: "none", sm: "0 1px 2px 0 rgb(0 0 0 / 0.05)" },
    backgroundImage: {},
  },
  plugins: [],
};

export default config;
