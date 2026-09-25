"use client";

import { Menu, Moon, Sun } from "lucide-react";
import { useTheme } from "next-themes";
import { useEffect, useState } from "react";

import { Brand } from "@/components/shell/brand";
import { NavList } from "@/components/shell/nav";
import { DISCLAIMER } from "@/components/shell/sidebar";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetDescription, SheetTitle, SheetTrigger } from "@/components/ui/sheet";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { useHealth } from "@/lib/queries";
import { cn } from "@/lib/utils";

/** Live API reachability: a dot plus text, checked every 30 s (colour is never the only signal). */
function ApiStatus() {
  const { data, isError, isPending } = useHealth();
  const state = isPending ? "checking" : isError || data?.status !== "ok" ? "degraded" : "ok";
  const label = { checking: "Checking API", ok: "API connected", degraded: "API degraded" }[state];
  const detail =
    state === "ok"
      ? `v${data?.version} · ${data?.environment}`
      : state === "degraded" && data
        ? Object.entries(data.checks)
            .filter(([, check]) => check.status === "down")
            .map(([name]) => `${name} is down`)
            .join(", ")
        : "The console cannot reach the API.";
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span tabIndex={0} role="status" className="flex items-center gap-2 rounded-md px-2 py-1 text-table text-muted">
          <span
            aria-hidden
            className={cn(
              "size-2 rounded-full",
              state === "ok" && "bg-approve",
              state === "degraded" && "bg-reject",
              state === "checking" && "bg-subtle motion-safe:animate-pulse",
            )}
          />
          <span className="hidden sm:inline">{label}</span>
        </span>
      </TooltipTrigger>
      <TooltipContent>{detail}</TooltipContent>
    </Tooltip>
  );
}

function ThemeToggle() {
  const { resolvedTheme, setTheme } = useTheme();
  // The theme is only known after mount (it comes from localStorage); render a stable icon until then.
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  const dark = mounted && resolvedTheme === "dark";
  return (
    <Button
      variant="ghost"
      size="icon-sm"
      aria-label={dark ? "Switch to light theme" : "Switch to dark theme"}
      onClick={() => setTheme(dark ? "light" : "dark")}
    >
      {dark ? <Sun aria-hidden /> : <Moon aria-hidden />}
    </Button>
  );
}

function MobileNav() {
  const [open, setOpen] = useState(false);
  return (
    <Sheet open={open} onOpenChange={setOpen}>
      <SheetTrigger asChild>
        <Button variant="ghost" size="icon-sm" aria-label="Open navigation" className="md:hidden">
          <Menu aria-hidden />
        </Button>
      </SheetTrigger>
      <SheetContent side="left">
        <SheetTitle className="sr-only">Navigation</SheetTitle>
        <SheetDescription className="sr-only">{DISCLAIMER}</SheetDescription>
        <Brand />
        <div className="px-3 py-2">
          <NavList onNavigate={() => setOpen(false)} />
        </div>
        <p className="mt-auto border-t border-border p-4 text-table text-muted">{DISCLAIMER}</p>
      </SheetContent>
    </Sheet>
  );
}

export function Topbar() {
  return (
    <header className="sticky top-0 z-30 flex h-14 items-center gap-2 border-b border-border bg-background px-4 md:px-8">
      <MobileNav />
      <div className="md:hidden">
        <Brand />
      </div>
      <div className="ml-auto flex items-center gap-1">
        <ApiStatus />
        <ThemeToggle />
      </div>
    </header>
  );
}
