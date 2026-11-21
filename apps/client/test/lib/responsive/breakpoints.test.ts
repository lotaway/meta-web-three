import { resolveBreakpoint } from '@/lib/responsive/breakpoints';

describe('responsive breakpoints', () => {
  it('maps widths below 640 to compact', () => {
    expect(resolveBreakpoint(0)).toBe('compact');
    expect(resolveBreakpoint(639)).toBe('compact');
  });

  it('maps widths 640 to 1023 to medium', () => {
    expect(resolveBreakpoint(640)).toBe('medium');
    expect(resolveBreakpoint(1023)).toBe('medium');
  });

  it('maps widths 1024 and above to expanded', () => {
    expect(resolveBreakpoint(1024)).toBe('expanded');
    expect(resolveBreakpoint(2560)).toBe('expanded');
  });
});
