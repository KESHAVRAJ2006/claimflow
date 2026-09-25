import { FileQuestion } from "lucide-react";
import Link from "next/link";

import { EmptyState } from "@/components/empty-state";
import { AppShell } from "@/components/shell/app-shell";
import { Button } from "@/components/ui/button";

export default function NotFound() {
  return (
    <AppShell>
      <EmptyState
        icon={FileQuestion}
        title="This page doesn't exist."
        action={
          <Button asChild variant="primary">
            <Link href="/dashboard">Go to dashboard</Link>
          </Button>
        }
      />
    </AppShell>
  );
}
