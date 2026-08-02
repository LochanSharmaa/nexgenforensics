/**
 * Owner auto-session: skip the sign-in screen during local development.
 *
 * The owner runs this app on their own machine and was signing in dozens of
 * times a day. Rather than removing authentication -- which would not work
 * anyway, because the API enforces it on every request and the UI would simply
 * collect 401s -- the app signs itself in with the operator account, silently,
 * once per page load. Every downstream request carries a real token, roles
 * still apply, and the audit trail still records who acted.
 *
 * SAFETY: gated on `import.meta.env.DEV`. Vite substitutes that literal at
 * build time, so in a production bundle this whole block is dead code and the
 * credential strings are eliminated -- they cannot be shipped even if the
 * variables are set in the build environment. Configure it in
 * frontend/.env.local, which is gitignored.
 */
const enabled =
  import.meta.env.DEV && Boolean(import.meta.env.VITE_OWNER_EMAIL && import.meta.env.VITE_OWNER_PASSWORD);

export const ownerSession = {
  enabled,
  email: enabled ? import.meta.env.VITE_OWNER_EMAIL : "",
  password: enabled ? import.meta.env.VITE_OWNER_PASSWORD : "",
  tenant: enabled ? import.meta.env.VITE_OWNER_TENANT || "" : "",
};
