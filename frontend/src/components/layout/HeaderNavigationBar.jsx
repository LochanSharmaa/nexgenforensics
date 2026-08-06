import { useEffect, useMemo, useRef, useState } from "react";
import { navGroups } from "../../constants/navigation";
import "./HeaderNavigationBar.css";

const directLinks = [
  { label: "About Us", href: "/about" },
  { label: "Contact", href: "/contact" },
];

const menuActions = [
  { label: "Try Online Demo", href: "/demo", kind: "primary" },
  { label: "Open iMATCH", href: "/workspace", kind: "outline" },
  { label: "Request Access", href: "/contact", kind: "outline" },
];

function isActiveHref(href, pathname) {
  return pathname === href || (href !== "/" && pathname.startsWith(`${href}/`));
}

function Chevron() {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M6 9l6 6 6-6" />
    </svg>
  );
}

/**
 * Phonexia-standard header. On desktop: logo left; Products / Solutions /
 * Resources with dropdown panels, About Us and Contact inline; a language
 * globe and one filled CTA pill on the right — all on a tall flat-white bar.
 * Under 980px the links collapse into a hamburger that opens the fullscreen
 * overlay menu, which is exactly what the reference site does on phones.
 */
export function HeaderNavigationBar() {
  const [menuOpen, setMenuOpen] = useState(false);
  const [openGroup, setOpenGroup] = useState(null);
  const navRef = useRef(null);
  const closeTimerRef = useRef(null);
  const pathname = window.location.pathname;

  const groups = useMemo(() => Object.entries(navGroups), []);

  useEffect(() => {
    const onKeyDown = (event) => {
      if (event.key === "Escape") {
        setMenuOpen(false);
        setOpenGroup(null);
      }
    };
    const onPointerDown = (event) => {
      if (navRef.current && !navRef.current.contains(event.target)) {
        setOpenGroup(null);
      }
    };
    document.addEventListener("keydown", onKeyDown);
    document.addEventListener("pointerdown", onPointerDown);
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      document.removeEventListener("pointerdown", onPointerDown);
      if (closeTimerRef.current) window.clearTimeout(closeTimerRef.current);
    };
  }, []);

  // The overlay owns the viewport while open; the page behind must not scroll.
  useEffect(() => {
    document.body.style.overflow = menuOpen ? "hidden" : "";
    return () => {
      document.body.style.overflow = "";
    };
  }, [menuOpen]);

  const cancelClose = () => {
    if (closeTimerRef.current) window.clearTimeout(closeTimerRef.current);
  };

  const scheduleClose = () => {
    cancelClose();
    closeTimerRef.current = window.setTimeout(() => setOpenGroup(null), 140);
  };

  const closeAll = () => {
    cancelClose();
    setMenuOpen(false);
    setOpenGroup(null);
  };

  const dropdownGroup = (key, label, content, extraClass = "") => (
    <div
      className={`nx-nav-group ${extraClass}`.trim()}
      key={key}
      onMouseEnter={() => {
        cancelClose();
        setOpenGroup(key);
      }}
      onMouseLeave={scheduleClose}
    >
      <button
        type="button"
        className={openGroup === key ? "open" : ""}
        aria-expanded={openGroup === key}
        aria-controls={`nx-dropdown-${key}`}
        onClick={() => setOpenGroup((value) => (value === key ? null : key))}
      >
        {label}
        <Chevron />
      </button>
      <div
        className={openGroup === key ? "nx-dropdown open" : "nx-dropdown"}
        id={`nx-dropdown-${key}`}
      >
        <div className="nx-dropdown-panel">{content}</div>
      </div>
    </div>
  );

  return (
    <>
      <nav className="nx-nav" ref={navRef} aria-label="NexGen Forensics primary navigation">
        <a href="/" className="nx-brand" onClick={closeAll}>
          <span className="nx-brand-mark" aria-hidden="true">
            <i />
            <i />
            <i />
          </span>
          NexGen Forensics
        </a>

        <div className="nx-nav-links">
          {groups.map(([key, group]) =>
            dropdownGroup(
              key,
              <span className={isActiveHref(group.href, pathname) ? "active" : ""}>
                {group.label}
              </span>,
              <>
                {group.items.map((item) => (
                  <a
                    href={item.href}
                    key={item.href}
                    className={isActiveHref(item.href, pathname) ? "active" : ""}
                    onClick={closeAll}
                  >
                    {item.title}
                  </a>
                ))}
                <a className="nx-dropdown-more" href={group.featured.href} onClick={closeAll}>
                  {group.featured.cta} →
                </a>
              </>,
            ),
          )}

          {directLinks.map((link) => (
            <a
              className={isActiveHref(link.href, pathname) ? "nx-nav-direct active" : "nx-nav-direct"}
              href={link.href}
              key={link.href}
              onClick={closeAll}
            >
              {link.label}
            </a>
          ))}
        </div>

        <div className="nx-nav-side">
          {dropdownGroup(
            "lang",
            <>
              <svg
                className="nx-globe"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.6"
                strokeLinecap="round"
                aria-hidden="true"
              >
                <circle cx="12" cy="12" r="9" />
                <path d="M3 12h18M12 3a13.5 13.5 0 0 1 0 18M12 3a13.5 13.5 0 0 0 0 18" />
              </svg>
              EN
            </>,
            <span className="nx-lang-current">English — current language</span>,
            "nx-nav-lang",
          )}

          <a className="nx-nav-cta" href="/demo" onClick={closeAll}>
            Try Online Demo
          </a>
        </div>

        <button
          type="button"
          className="nx-menu-toggle"
          aria-label={menuOpen ? "Close navigation menu" : "Open navigation menu"}
          aria-expanded={menuOpen}
          aria-controls="nx-menu-overlay"
          onClick={() => setMenuOpen((value) => !value)}
        >
          <svg
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            {menuOpen ? <path d="M6 6l12 12M18 6L6 18" /> : <path d="M4 8h16M4 16h16" />}
          </svg>
        </button>
      </nav>

      <div
        className={menuOpen ? "nx-menu-overlay open" : "nx-menu-overlay"}
        id="nx-menu-overlay"
        role="dialog"
        aria-modal="true"
        aria-label="Site navigation"
      >
        <div className="nx-menu-scroll">
          <div className="nx-menu-inner">
            {groups.map(([key, group], index) => (
              <section
                className={openGroup === key ? "nx-menu-group open" : "nx-menu-group"}
                key={key}
                style={{ "--menu-index": index }}
              >
                <button
                  type="button"
                  aria-expanded={openGroup === key}
                  aria-controls={`nx-menu-panel-${key}`}
                  onClick={() => setOpenGroup((value) => (value === key ? null : key))}
                >
                  <span className={isActiveHref(group.href, pathname) ? "active" : ""}>
                    {group.label}
                  </span>
                  <Chevron />
                </button>

                <div className="nx-menu-panel" id={`nx-menu-panel-${key}`}>
                  <div>
                    <p className="nx-menu-subtitle">{group.subtitle}</p>
                    <ul>
                      {group.items.map((item) => (
                        <li key={item.href}>
                          <a
                            href={item.href}
                            className={isActiveHref(item.href, pathname) ? "active" : ""}
                            onClick={closeAll}
                          >
                            {item.title}
                          </a>
                        </li>
                      ))}
                      <li>
                        <a className="nx-menu-more" href={group.featured.href} onClick={closeAll}>
                          {group.featured.cta} →
                        </a>
                      </li>
                    </ul>
                  </div>
                </div>
              </section>
            ))}

            {directLinks.map((link, index) => (
              <a
                className={
                  isActiveHref(link.href, pathname) ? "nx-menu-direct active" : "nx-menu-direct"
                }
                href={link.href}
                key={link.href}
                style={{ "--menu-index": groups.length + index }}
                onClick={closeAll}
              >
                {link.label}
              </a>
            ))}

            <div
              className="nx-menu-actions"
              style={{ "--menu-index": groups.length + directLinks.length }}
            >
              {menuActions.map((action) => (
                <a
                  key={action.href + action.label}
                  className={`nx-menu-pill ${action.kind}`}
                  href={action.href}
                  onClick={closeAll}
                >
                  {action.label}
                </a>
              ))}
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

export default HeaderNavigationBar;
