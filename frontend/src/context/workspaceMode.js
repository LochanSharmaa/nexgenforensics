/**
 * Which landing experience a signed-in user prefers.
 *
 * IMPORTANT — this is NOT a permission.
 *
 * The backend already has a role hierarchy (`investigator` < `supervisor` <
 * `admin`) which is enforced server-side on every request and surfaced through
 * `useAuth().hasRole`. This value is a different thing entirely: a per-browser
 * UI preference deciding which page a user lands on after signing in.
 *
 * Conflating the two would be a real bug. Choosing "Individual" must not imply
 * reduced privileges, and choosing "Investigator" must not grant any. A user
 * who picks "Individual" and then navigates to /workspace still gets in if the
 * server says their role allows it — and still gets refused if it does not.
 *
 * Stored in localStorage rather than on the account because it is a device
 * preference, and because persisting it server-side would make it look like an
 * authorization attribute to anyone reading the user record later.
 */

const KEY = "nx.workspaceMode";
export const MODES = ["investigator", "individual"];

export function getWorkspaceMode() {
  try {
    const value = window.localStorage.getItem(KEY);
    return MODES.includes(value) ? value : null;
  } catch {
    // Private browsing or blocked storage: behave as "not chosen yet" rather
    // than throwing. The user sees the choice screen again, which is harmless.
    return null;
  }
}

export function setWorkspaceMode(mode) {
  if (!MODES.includes(mode)) return;
  try {
    window.localStorage.setItem(KEY, mode);
  } catch {
    /* preference simply does not persist; the flow still works */
  }
}

export function clearWorkspaceMode() {
  try {
    window.localStorage.removeItem(KEY);
  } catch {
    /* nothing to do */
  }
}

/**
 * Where a signed-in user should land.
 *
 * A specific workspace destination the user was already heading for wins over
 * the mode preference — otherwise being bounced through login would silently
 * drop the page they asked for.
 */
export function destinationFor(mode, from) {
  const pathname = from?.pathname;
  if (pathname && pathname.startsWith("/workspace")) return pathname;
  return mode === "individual" ? "/compare" : "/workspace";
}
