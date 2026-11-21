import { useWindowDimensions } from 'react-native';
import { resolveBreakpoint, type Breakpoint } from '@/lib/responsive/breakpoints';

export interface ResponsiveMetrics {
  width: number;
  height: number;
  isLandscape: boolean;
  breakpoint: Breakpoint;
  contentPadding: number;
  columns: number;
}

export function useResponsive(): ResponsiveMetrics {
  const { width, height } = useWindowDimensions();
  const breakpoint = resolveBreakpoint(width);
  return {
    width,
    height,
    isLandscape: width > height,
    breakpoint,
    contentPadding: breakpoint === 'compact' ? 12 : 16,
    columns: breakpoint === 'expanded' ? 4 : breakpoint === 'medium' ? 2 : 1,
  };
}

export { resolveBreakpoint, type Breakpoint } from '@/lib/responsive/breakpoints';
