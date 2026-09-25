/**
 * Motion tokens. Every animation uses these, so the whole console moves the same way: 150–300 ms, one easing curve,
 * 60 ms stagger. <MotionConfig reducedMotion="user"> in app/providers.tsx turns transforms off for users who ask
 * their OS for reduced motion.
 */
import type { Transition, Variants } from "framer-motion";

export const EASE: [number, number, number, number] = [0.22, 1, 0.36, 1];
export const DURATION = { fast: 0.15, base: 0.2, slow: 0.3 } as const;
export const STAGGER = 0.06;

export const baseTransition: Transition = { duration: DURATION.base, ease: EASE };

/** Fade in while rising 4px: used for list items and cards as they appear. */
export const fadeUp: Variants = {
  hidden: { opacity: 0, y: 4 },
  show: { opacity: 1, y: 0, transition: baseTransition },
};

/** Parent variant that staggers its children by 60 ms. */
export const stagger: Variants = {
  hidden: {},
  show: { transition: { staggerChildren: STAGGER } },
};
