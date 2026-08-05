import { NavLink, Outlet } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { EngineBanner } from "./components/EngineBanner";
import "./workspace.css";

const LINKS = [
  { to: "/workspace", label: "Cases", end: true },
  { to: "/workspace/search", label: "Face search" },
  { to: "/workspace/enhance", label: "Enhance" },
  { to: "/workspace/verify", label: "1:1 compare" },
  { to: "/workspace/batch-compare", label: "Reference vs set" },
  { to: "/workspace/enrol", label: "Enrol subject", minRole: "supervisor" },
  { to: "/workspace/audit", label: "Audit trail" },
];

export function WorkspaceLayout() {
  const { user, hasRole } = useAuth();

  return (
    <div className="wk-shell">
      <header className="wk-topbar">
        <div className="wk-brand">
          NexGen iMATCH
          <small>Investigator workspace</small>
        </div>

        <nav className="wk-nav" aria-label="Workspace sections">
          {LINKS.filter((link) => !link.minRole || hasRole(link.minRole)).map((link) => (
            <NavLink
              key={link.to}
              to={link.to}
              end={link.end}
              className={({ isActive }) => (isActive ? "active" : "")}
            >
              {link.label}
            </NavLink>
          ))}
        </nav>

        <div className="wk-user">
          <span>
            <b>{user?.full_name || user?.email || "Local operator"}</b>
            {user?.role}
          </span>
        </div>
      </header>

      <main className="wk-main">
        <EngineBanner />
        <Outlet />
      </main>
    </div>
  );
}
