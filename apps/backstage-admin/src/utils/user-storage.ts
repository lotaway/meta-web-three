/** Storage key used by the persisted user store. */
export const USER_PERSIST_KEY = 'user'

/** Minimal Web Storage contract needed to purge a stored credential. */
export interface UserStorage {
  getItem(key: string): string | null
  setItem(key: string, value: string): void
  removeItem(key: string): void
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

/** Null means the stored value is unusable and must be discarded. */
function parseState(raw: string): Record<string, unknown> | null {
  try {
    const value: unknown = JSON.parse(raw)
    return isRecord(value) ? value : null
  } catch {
    return null
  }
}

function stripPassword(state: Record<string, unknown>): boolean {
  const userInfo = state.userInfo
  const hasNested = isRecord(userInfo) && 'password' in userInfo
  if (isRecord(userInfo) && hasNested) delete userInfo.password
  const hasTopLevel = 'password' in state
  if (hasTopLevel) delete state.password
  return hasNested || hasTopLevel
}

/**
 * Remove a password persisted by older builds while keeping the session intact.
 * Returns true when the stored value was rewritten or discarded.
 */
export function purgePersistedPassword(
  storage: UserStorage,
  key: string = USER_PERSIST_KEY,
): boolean {
  const raw = storage.getItem(key)
  if (raw === null) return false
  const state = parseState(raw)
  if (state === null) {
    // Damaged state cannot be repaired; drop it so no credential survives.
    storage.removeItem(key)
    return true
  }
  if (!stripPassword(state)) return false
  storage.setItem(key, JSON.stringify(state))
  return true
}
