import { redirect } from "next/navigation";

/** The console has no landing page; "/" opens the dashboard. */
export default function Home() {
  redirect("/dashboard");
}
