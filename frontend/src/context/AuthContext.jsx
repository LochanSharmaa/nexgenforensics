import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { fetchCurrentUser, login as apiLogin, logout as apiLogout, tokenStore } from "../services/imatchApi";
import { ownerSession } from "../config/ownerSession";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  // Resolve the operator on mount. /api/auth/me answers even without a stored
  // token: the backend runs in single-user mode and treats a credential-less
  // request as the local owner, so this is the whole "sign-in".
  useEffect(() => {
    let cancelled = false;

    // Fallback for a deployment running with NEXGEN_SINGLE_USER=false, where
    // /me requires a real session. See config/ownerSession.js -- compiled out
    // of production builds.
    async function signInAsOwner() {
      if (!ownerSession.enabled) return;
      try {
        const profile = await apiLogin({
          email: ownerSession.email,
          password: ownerSession.password,
          tenant: ownerSession.tenant,
        });
        if (!cancelled) setUser(profile);
      } catch (error) {
        // Left signed out rather than retried: a wrong password here would
        // otherwise walk the account into its brute-force lockout.
        console.warn("Owner auto-session failed.", error);
      }
    }

    async function restore() {
      try {
        const profile = await fetchCurrentUser();
        if (!cancelled) setUser(profile);
      } catch {
        tokenStore.clear();
        await signInAsOwner();
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    restore();
    return () => {
      cancelled = true;
    };
  }, []);

  const signIn = useCallback(async (credentials) => {
    const profile = await apiLogin(credentials);
    setUser(profile);
    return profile;
  }, []);

  const signOut = useCallback(async () => {
    await apiLogout();
    setUser(null);
  }, []);

  const value = useMemo(
    () => ({
      user,
      loading,
      signIn,
      signOut,
      isAuthenticated: Boolean(user),
      hasRole: (minimum) => {
        const rank = { investigator: 1, supervisor: 2, admin: 3 };
        return Boolean(user) && (rank[user.role] || 0) >= (rank[minimum] || 0);
      },
    }),
    [user, loading, signIn, signOut],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used inside an AuthProvider.");
  }
  return context;
}
