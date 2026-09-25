"use client";

import { Info, PanelLeftClose, PanelLeftOpen } from "lucide-react";

import { Brand } from "@/components/shell/brand";
import { NavList } from "@/components/shell/nav";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

export const DISCLAIMER =
  "Decisions shown here are AI-assisted recommendations that require human review. They are not automated determinations.";

/** Desktop sidebar: 220px, collapsible to a 64px icon rail. Hidden below the md breakpoint (the drawer takes over). */
export function Sidebar({ collapsed, onToggle }: { collapsed: boolean; onToggle: () => void }) {
  return (
    <aside
      className={cn(
        "sticky top-0 hidden h-screen shrink-0 flex-col border-r border-border bg-surface transition-[width] duration-200 ease-out-expo md:flex",
        collapsed ? "w-rail" : "w-sidebar",
      )}
    >
      <Brand collapsed={collapsed} />
      <div className="flex-1 overflow-y-auto px-3 py-2">
        <NavList collapsed={collapsed} />
      </div>
      <div className="flex flex-col gap-2 border-t border-border p-3">
        {collapsed ? (
          <Tooltip>
            <TooltipTrigger asChild>
              <span tabIndex={0} className="flex justify-center rounded-md py-1 text-subtle" aria-label={DISCLAIMER}>
                <Info aria-hidden className="size-4" />
              </span>
            </TooltipTrigger>
            <TooltipContent side="right" className="max-w-60">
              {DISCLAIMER}
            </TooltipContent>
          </Tooltip>
        ) : (
          <p className="flex gap-2 text-table text-muted">
            <Info aria-hidden className="mt-0.5 size-3.5 shrink-0 text-subtle" />
            <span>AI-assisted recommendations. Every decision requires human review.</span>
          </p>
        )}
        <Button
          variant="ghost"
          size={collapsed ? "icon-sm" : "sm"}
          onClick={onToggle}
          aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          className={cn(!collapsed && "justify-start")}
        >
          {collapsed ? <PanelLeftOpen aria-hidden /> : <PanelLeftClose aria-hidden />}
          {collapsed ? null : "Collapse"}
        </Button>
      </div>
    </aside>
  );
}
