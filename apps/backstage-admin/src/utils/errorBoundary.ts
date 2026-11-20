export interface NormalizedError {
  message: string;
}

export function normalizeError(error: unknown): NormalizedError {
  if (error instanceof Error) {
    return { message: error.message || error.name };
  }
  if (typeof error === 'string' && error.length > 0) {
    return { message: error };
  }
  return { message: 'Unexpected error' };
}
