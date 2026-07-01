import { useEffect, useMemo, useRef, useState } from "react";
import { navGroups } from "../../constants/navigation";
import "./HeaderNavigationBar.css";

const directLinks = [
  { label: "About Us", href: "/about" },
  { label: "Contact", href: "/contact" },
];

function isActiveHref(href, pathname) {
  return pathname === href || (href !== "/" && pathname.startsWith(`${href}/`));
}

export function HeaderNavigationBar() {
  const [openGroup, setOpenGroup] = useState(null);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const navRef = useRef(null);
  const pathname = window.location.pathname;

  const groups = useMemo(() => Object.entries(navGroups), []);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 12);
    const onPointerDown = (event) => {
      if (navRef.current && !navRef.current.contains(event.target)) {
        setOpenGroup(null);
      }
    };
    const onKeyDown = (event) => {
      if (event.key === "Escape") {
        setOpenGroup(null);
        setMobileOpen(false);
      }
    };

    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    document.addEventListener("pointerdown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("scroll", onScroll);
      document.removeEventListener("pointerdown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, []);

  const closeAll = () => {
    setOpenGroup(null);
    setMobileOpen(false);
  };

  return (
    <nav
      className={scrolled ? "nx-nav scrolled" : "nx-nav"}
      ref={navRef}
      onMouseLeave={() => setOpenGroup(null)}
      aria-label="NexGen Forensics primary navigation"
    >
      <a href="/" className="nx-brand" onClick={closeAll}>
        NexGen Forensics
      </a>

      <button
        className="nx-mobile-toggle"
        type="button"
        aria-label="Open navigation menu"
        aria-expanded={mobileOpen}
        onClick={() => setMobileOpen((value) => !value)}
      >
        <span />
        <span />
      </button>

      <div className={mobileOpen ? "nx-nav-menu open" : "nx-nav-menu"}>
        <div className="nx-nav-links">
          {groups.map(([key, group]) => (
            <div className="nx-nav-group" key={key}>
              <button
                type="button"
                className={isActiveHref(group.href, pathname) ? "active" : ""}
                aria-expanded={openGroup === key}
                aria-controls={`mega-${key}`}
                onMouseEnter={() => setOpenGroup(key)}
                onFocus={() => setOpenGroup(key)}
                onClick={() => setOpenGroup((value) => (value === key ? null : key))}
              >
                {group.label}
              </button>
              <MegaMenu group={group} groupKey={key} open={openGroup === key} onNavigate={closeAll} />
            </div>
          ))}

          {directLinks.map((link) => (
            <a
              className={isActiveHref(link.href, pathname) ? "active" : ""}
              href={link.href}
              key={link.href}
              onClick={closeAll}
            >
              {link.label}
            </a>
          ))}
        </div>

        <div className="nx-nav-actions">
          <a className="nx-nav-login" href="/login" onClick={closeAll}>
            Login
          </a>
          <a className="nx-nav-secondary" href="/contact" onClick={closeAll}>
            Request Access
          </a>
          <a className="nx-nav-cta" href="/demo" onClick={closeAll}>
            Try Online Demo
          </a>
        </div>
      </div>
    </nav>
  );
}

function MegaMenu({ group, groupKey, open, onNavigate }) {
  return (
    <div className={open ? "nx-mega open" : "nx-mega"} id={`mega-${groupKey}`}>
      <div className="nx-mega-panel">
        <div className="nx-mega-intro">
          <span>{group.title}</span>
          <p>{group.subtitle}</p>
        </div>

        <div className="nx-mega-grid">
          <div className="nx-mega-cards">
            {group.items.map((item, index) => (
              <a
                className="nx-mega-card"
                href={item.href}
                key={item.href}
                onClick={onNavigate}
                style={{ "--mega-index": index }}
              >
                <i aria-hidden="true" />
                <span>{item.category || group.title}</span>
                <strong>{item.title}</strong>
                <p>{item.description}</p>
                <b aria-hidden="true">→</b>
              </a>
            ))}
          </div>

          <a className="nx-mega-feature" href={group.featured.href} onClick={onNavigate}>
            <span>Featured</span>
            <strong>{group.featured.title}</strong>
            <p>{group.featured.text}</p>
            <em>{group.featured.cta} →</em>
          </a>
        </div>
      </div>
    </div>
  );
}

export default HeaderNavigationBar;
