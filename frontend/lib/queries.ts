"use client";

/**
 * TanStack Query hooks. Query keys are built in one place so a mutation can invalidate exactly what it changed.
 */
import { keepPreviousData, useQuery } from "@tanstack/react-query";

import { api, type ClaimListParams } from "@/lib/api";

export const queryKeys = {
  health: ["health"] as const,
  metrics: (days: number) => ["metrics", days] as const,
  claims: (params: ClaimListParams) => ["claims", params] as const,
  claim: (id: string) => ["claim", id] as const,
};

/** API reachability for the topbar dot; rechecked every 30 s. */
export function useHealth() {
  return useQuery({ queryKey: queryKeys.health, queryFn: api.health, refetchInterval: 30_000, retry: false });
}

export function useMetrics(days = 30) {
  return useQuery({ queryKey: queryKeys.metrics(days), queryFn: () => api.metrics(days), refetchInterval: 60_000 });
}

/** A page of claims. keepPreviousData keeps the current rows visible while the next page or sort loads. */
export function useClaims(params: ClaimListParams) {
  return useQuery({
    queryKey: queryKeys.claims(params),
    queryFn: () => api.listClaims(params),
    placeholderData: keepPreviousData,
  });
}

export function useClaim(id: string) {
  return useQuery({ queryKey: queryKeys.claim(id), queryFn: () => api.getClaim(id) });
}
