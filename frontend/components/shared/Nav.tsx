"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Sparkles, LayoutDashboard, Settings, CreditCard } from "lucide-react";

export default function Nav() {
  const path = usePathname();

  // Extract project id if inside a project
  const projectMatch = path.match(/\/projects\/(\d+)/);
  const projectId = projectMatch ? projectMatch[1] : null;

  const globalLinks = [
    { href: "/dashboard", label: "Dashboard", icon: <LayoutDashboard size={15} /> },
    { href: "/settings", label: "Settings", icon: <Settings size={15} /> },
  ];

  const projectLinks = projectId
    ? [
        { href: `/projects/${projectId}/brief`,     label: "Brief"     },
        { href: `/projects/${projectId}/concepts`,   label: "Board"     },
        { href: `/projects/${projectId}/missing`,    label: "Gaps"      },
        { href: `/projects/${projectId}/images`,     label: "Images"    },
        { href: `/projects/${projectId}/genealogy`,  label: "Genealogy" },
        { href: `/projects/${projectId}/saved`,      label: "Saved"     },
      ]
    : [];

  return (
    <nav className="nav" style={{ flexDirection: "column", padding: "0", alignItems: "stretch" }}>
      {/* Top bar */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "14px 32px", width: "100%" }}>
        <Link href="/" className="nav-brand" style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <Sparkles size={16} style={{ color: "#818cf8" }} />
          SIFS Creative Reasoning Engine
        </Link>

        <div className="nav-links">
          {globalLinks.map((l) => (
            <Link
              key={l.href}
              href={l.href}
              className="btn-ghost"
              style={path.startsWith(l.href) ? { color: "var(--text-primary)", background: "rgba(255,255,255,0.05)" } : {}}
            >
              {l.icon}
              {l.label}
            </Link>
          ))}
          {path.startsWith("/settings") && (
            <Link
              href="/settings/billing"
              className="btn-ghost"
              style={{ fontSize: "13px", ...(path.startsWith("/settings/billing") ? { color: "var(--text-primary)", background: "rgba(255,255,255,0.05)" } : {}) }}
            >
              <CreditCard size={15} /> Billing
            </Link>
          )}
          <Link href="/projects/new" className="btn-primary" style={{ padding: "8px 18px", fontSize: "13px" }}>
            <Sparkles size={13} /> New Project
          </Link>
        </div>
      </div>

      {/* Project sub-nav */}
      {projectLinks.length > 0 && (
        <div style={{
          display: "flex",
          gap: "2px",
          padding: "0 28px",
          borderTop: "1px solid var(--border)",
          background: "rgba(0,0,0,0.2)",
          overflowX: "auto",
        }}>
          {projectLinks.map(l => {
            const active = path === l.href || path.startsWith(l.href + "/");
            return (
              <Link
                key={l.href}
                href={l.href}
                style={{
                  padding: "8px 14px",
                  fontSize: "13px",
                  fontWeight: active ? 600 : 400,
                  color: active ? "var(--text-primary)" : "var(--text-muted)",
                  textDecoration: "none",
                  borderBottom: active ? "2px solid var(--accent)" : "2px solid transparent",
                  whiteSpace: "nowrap",
                  transition: "color 0.15s, border-color 0.15s",
                }}
              >
                {l.label}
              </Link>
            );
          })}
        </div>
      )}
    </nav>
  );
}
