import { useCallback } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

/**
 * Gate a public-demo action behind sign-in.
 *
 * The marketing demo used to call the API directly whether or not anyone was
 * signed in. Unauthenticated calls come back 401 — or, when the backend is not
 * running at all, as a connection failure — and either way the raw message was
 * rendered to a visitor on a public page. That is the wrong failure for this
 * surface: the demo should ask people to sign in, not show them a stack of
 * plumbing.
 *
 * Returns a function to call FIRST in an action handler:
 *
 *     if (requireLogin({ panel: "batch" })) return;   // redirected; do nothing
 *
 * `intent` is carried in navigation state so the login flow can send the user
 * somewhere sensible afterwards. Selected FILES are deliberately not carried:
 * File objects do not survive a route change, and stashing image bytes in
 * storage to rebuild them would put biometric data somewhere it should not be
 * for a convenience feature.
 */
export function useLoginGate() {
  const { isAuthenticated } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  return useCallback(
    (intent) => {
      if (isAuthenticated) return false;
      navigate("/login", { state: { from: location, intent } });
      return true;
    },
    [isAuthenticated, navigate, location],
  );
}
