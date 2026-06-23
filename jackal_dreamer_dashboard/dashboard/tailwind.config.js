/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: {
          base: "#0b0d10",
          panel: "#13171c",
          raised: "#1a1f25",
        },
        border: {
          subtle: "#252b33",
          strong: "#3a4350",
        },
        text: {
          primary: "#e3e8ee",
          muted: "#7a8590",
        },
        accent: {
          lidar: "#22d3ee",
          dreamer: "#3b82f6",
          expert: "#f59e0b",
          warning: "#fbbf24",
          danger: "#ef4444",
          ok: "#10b981",
        },
      },
      fontFamily: {
        mono: [
          "JetBrains Mono",
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "Consolas",
          "monospace",
        ],
        sans: [
          "Inter",
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "sans-serif",
        ],
      },
      fontSize: {
        "header": ["11px", { lineHeight: "14px", letterSpacing: "0.12em" }],
        "metric": ["14px", { lineHeight: "18px" }],
        "label": ["12px", { lineHeight: "16px" }],
        "pill": ["11px", { lineHeight: "14px" }],
      },
      boxShadow: {
        panel: "inset 0 0 0 1px #252b33",
      },
    },
  },
  plugins: [],
};
