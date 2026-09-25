"use client";

import { BookOpenText, FilePlus2, Inbox, LayoutDashboard, type LucideIcon } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";

import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

interface NavItem {
  href: string;
  label: string;
  icon: LucideIcon;
  /** Decides whether the item is highlighted for the current path. */
  isActive: (pathname: string) => boolean;
}

export const NAV_ITEMS: NavItem[] = [
  { href: "/dashboard", label: "Dashboard", icon: LayoutDashboard, isActive: (p) => p.startsWith("/dashboard") },
  // /claims/new belongs to "Submit claim", not to the queue.
  {
    href: "/claims",
    label: "Claims",
    icon: Inbox,
    isActive: (p) => p.startsWith("/claims") && !p.startsWith("/claims/new"),
  },
  { href: "/claims/new", label: "Submit claim", icon: FilePlus2, isActive: (p) => p.startsWith("/claims/new") },
  { href: "/policies", label: "Policies", icon: BookOpenText, isActive: (p) => p.startsWith("/policies") },
];

/** The navigation list, shared by the desktop sidebar and the mobile drawer. */
export function NavList({ collapsed = false, onNavigate }: { collapsed?: boolean; onNavigate?: () => void }) {
  const pathname = usePathname();
  return (
    <nav aria-label="Main">
      <ul className="flex flex-col gap-1">
        {NAV_ITEMS.map(({ href, label, icon: Icon, isActive }) => {
          const active = isActive(pathname);
          const link = (
            <Link
              href={href}
              onClick={onNavigate}
              aria-current={active ? "page" : undefined}
              aria-label={collapsed ? label : undefined}
              className={cn(
                "flex h-9 items-center gap-3 rounded-md px-3 text-body font-medium transition-colors duration-150",
                collapsed && "justify-center px-0",
                active ? "bg-surface-muted text-foreground" : "text-muted hover:bg-surface-muted hover:text-foreground",
              )}
            >
              <Icon aria-hidden className={cn("size-4 shrink-0", active && "text-foreground")} />
              {collapsed ? null : <span className="truncate">{label}</span>}
            </Link>
          );
          return (
            <li key={href}>
              {collapsed ? (
                // In the icon rail the label moves into a tooltip.
                <Tooltip>
                  <TooltipTrigger asChild>{link}</TooltipTrigger>
                  <TooltipContent side="right">{label}</TooltipContent>
                </Tooltip>
              ) : (
                link
              )}
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
