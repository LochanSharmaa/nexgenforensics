"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Sparkles, LayoutDashboard, Settings } from "lucide-react";

export default function Nav() {
  const path = usePathname();

  const links = [
    { href: "/dashboard", label: "Dashboard", icon: <LayoutDashboard size={15} /> },
    { href: "/settings", label: "Settings", icon: <Settings size={15} /> },
  ];

  return (
    <nav className="nav">
      <Link href="/" className="nav-brand" style={{ display: "flex", alignItems: "center", gap: "8px" }}>
        <Sparkles size={16} style={{ color: "#818cf8" }} />
        SIFS Creative Reasoning Engine
      </Link>

      <div className="nav-links">
        {links.map((l) => (
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
        <Link href="/projects/new" className="btn-primary" style={{ padding: "8px 18px", fontSize: "13px" }}>
          <Sparkles size={13} /> New Project
        </Link>
      </div>
    </nav>
  );
}
