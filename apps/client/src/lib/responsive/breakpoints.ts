export type Breakpoint = 'compact' | 'medium' | 'expanded';

const COMPACT_WIDTH = 640;
const EXPANDED_WIDTH = 1024;

export function resolveBreakpoint(width: number): Breakpoint {
  if (width < COMPACT_WIDTH) return 'compact';
  if (width < EXPANDED_WIDTH) return 'medium';
  return 'expanded';
}
