"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MotionConfig } from "framer-motion";
import { ThemeProvider } from "next-themes";
import { useState, type ReactNode } from "react";

import { TooltipProvider } from "@/components/ui/tooltip";
import { ApiError } from "@/lib/api";

function makeQueryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: {
        // Data counts as fresh for 10 s, so switching pages doesn't refetch everything at once.
        staleTime: 10_000,
        // Retrying a 4xx (bad request, not found, unauthorized) can't succeed; only retry network and 5xx errors.
        retry: (failureCount, error) =>
          failureCount < 2 && !(error instanceof ApiError && error.status >= 400 && error.status < 500),
        refetchOnWindowFocus: false,
      },
    },
  });
}

export function Providers({ children }: { children: ReactNode }) {
  // One client per browser session. useState (not a module variable) keeps server renders isolated per request.
  const [queryClient] = useState(makeQueryClient);
  // Light-first (the brief): light by default; the topbar toggle switches to dark and is remembered.
  return (
    <ThemeProvider attribute="class" defaultTheme="light" enableSystem={false} disableTransitionOnChange>
      <QueryClientProvider client={queryClient}>
        {/* reducedMotion="user": framer-motion drops transform animations when the OS asks for less motion. */}
        <MotionConfig reducedMotion="user">
          <TooltipProvider delayDuration={300}>{children}</TooltipProvider>
        </MotionConfig>
      </QueryClientProvider>
    </ThemeProvider>
  );
}
