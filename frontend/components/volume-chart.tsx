"use client";

import { useTheme } from "next-themes";
import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis, type TooltipProps } from "recharts";

import { formatDate } from "@/lib/format";
import type { VolumePoint } from "@/lib/types";

/**
 * Daily claim volume, stacked by triage state.
 *
 * Palette (validated with the dataviz skill's validator, light and dark surfaces): emerald-600 / amber-600 / zinc-500.
 * The series ARE claim states, so status hues are the right encoding; "not yet triaged" is deliberately neutral.
 * Protan separation of emerald vs amber is 7.9 (floor band), so identity never relies on colour alone: 2px surface
 * gaps between stacked segments, an always-visible labelled legend, a tooltip, and a screen-reader table.
 */
const SERIES = [
  { key: "auto_decided", label: "Auto-decided", color: "#059669" },
  { key: "escalated", label: "Escalated", color: "#D97706" },
  { key: "other", label: "Not yet triaged", color: "#71717A" },
] as const;

interface Row {
  date: string;
  auto_decided: number;
  escalated: number;
  other: number;
  submitted: number;
}

/** Axis and grid colours follow the theme; SVG attributes can't read CSS variables, so resolve them in JS. */
function useChartTheme() {
  const { resolvedTheme } = useTheme();
  const [theme, setTheme] = useState({ surface: "#FFFFFF", border: "#E4E4E7", muted: "#71717A" });
  useEffect(() => {
    const css = getComputedStyle(document.documentElement);
    const read = (name: string) => `rgb(${css.getPropertyValue(name).trim().split(/\s+/).join(", ")})`;
    setTheme({ surface: read("--surface"), border: read("--border"), muted: read("--muted") });
  }, [resolvedTheme]);
  return theme;
}

function ChartTooltip({ active, payload }: TooltipProps<number, string>) {
  const row = payload?.[0]?.payload as Row | undefined;
  if (!active || !row) return null;
  return (
    <div className="rounded-md border border-border bg-surface px-3 py-2 shadow-sm">
      <p className="text-table font-medium text-foreground">{formatDate(row.date)}</p>
      <ul className="mt-1 flex flex-col gap-0.5">
        {SERIES.map((series) => (
          <li key={series.key} className="flex items-center gap-2 text-table text-muted">
            <span aria-hidden className="size-2 rounded-sm" style={{ background: series.color }} />
            {series.label}
            <span className="ml-auto pl-4 font-mono text-foreground">{row[series.key]}</span>
          </li>
        ))}
        <li className="mt-0.5 flex border-t border-border pt-0.5 text-table text-muted">
          Total<span className="ml-auto font-mono text-foreground">{row.submitted}</span>
        </li>
      </ul>
    </div>
  );
}

export function VolumeChart({ volume }: { volume: VolumePoint[] }) {
  const theme = useChartTheme();
  const rows: Row[] = volume.map((point) => ({
    date: point.date,
    auto_decided: point.auto_decided,
    escalated: point.escalated,
    other: Math.max(0, point.submitted - point.auto_decided - point.escalated),
    submitted: point.submitted,
  }));
  return (
    <figure className="flex flex-col gap-4">
      <ul className="flex flex-wrap gap-4" aria-label="Legend">
        {SERIES.map((series) => (
          <li key={series.key} className="flex items-center gap-2 text-table text-muted">
            <span aria-hidden className="size-2.5 rounded-sm" style={{ background: series.color }} />
            {series.label}
          </li>
        ))}
      </ul>
      <div className="h-60" aria-hidden>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={rows} margin={{ top: 4, right: 4, bottom: 0, left: -24 }} barCategoryGap="20%">
            <CartesianGrid vertical={false} stroke={theme.border} strokeDasharray="3 3" />
            <XAxis
              dataKey="date"
              tickFormatter={(value: string) => formatDate(value).slice(0, 6)}
              tick={{ fill: theme.muted, fontSize: 12 }}
              axisLine={false}
              tickLine={false}
              minTickGap={24}
            />
            <YAxis allowDecimals={false} tick={{ fill: theme.muted, fontSize: 12 }} axisLine={false} tickLine={false} />
            <Tooltip content={<ChartTooltip />} cursor={{ fill: theme.border, opacity: 0.4 }} />
            {SERIES.map((series, index) => (
              <Bar
                key={series.key}
                dataKey={series.key}
                stackId="volume"
                fill={series.color}
                // 2px surface-coloured stroke = the gap between stacked segments.
                stroke={theme.surface}
                strokeWidth={2}
                radius={index === SERIES.length - 1 ? [4, 4, 0, 0] : 0}
                isAnimationActive={false}
              />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
      <table className="sr-only">
        <caption>Claims per day by triage state</caption>
        <thead>
          <tr>
            <th scope="col">Date</th>
            {SERIES.map((series) => (
              <th key={series.key} scope="col">
                {series.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.date}>
              <th scope="row">{formatDate(row.date)}</th>
              {SERIES.map((series) => (
                <td key={series.key}>{row[series.key]}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </figure>
  );
}
