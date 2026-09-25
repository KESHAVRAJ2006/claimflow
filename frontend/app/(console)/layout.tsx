import type { ReactNode } from "react";

import { AppShell } from "@/components/shell/app-shell";

/** Every console page shares the sidebar + topbar shell. The (console) group adds no URL segment. */
export default function ConsoleLayout({ children }: { children: ReactNode }) {
  return <AppShell>{children}</AppShell>;
}
