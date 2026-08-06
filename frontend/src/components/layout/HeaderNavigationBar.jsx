import { useEffect, useMemo, useState } from "react";
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

/**
 * Phonexia-standard header: the bar itself carries only the wordmark and a
 * menu glyph — at every viewport width, not just phones. All navigation lives
 * in a fullscreen frosted overlay so the bar never has to compress, reflow or
 * hide links as the viewport narrows.
 */
export function HeaderNavigationBar() {
  const [menuOpen, setMenuOpen] = useState(false);
  const [openGroup, setOpenGroup] = useState(null);
  const pathname = window.location.pathname;

  const groups = useMemo(() => Object.entries(navGroups), []);

  useEffect(() => {
    const onKeyDown = (event) => {
      if (event.key === "Escape") setMenuOpen(false);
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, []);

  // The overlay owns the viewport while open; the page behind must not scroll.
  useEffect(() => {
    document.body.style.overflow = menuOpen ? "hidden" : "";
    return () => {
      document.body.style.overflow = "";
    };
  }, [menuOpen]);

  const closeMenu = () => {
    setMenuOpen(false);
    setOpenGroup(null);
  };

  return (
    <>
      <nav className="nx-nav" aria-label="NexGen Forensics primary navigation">
        <a href="/" className="nx-brand" onClick={closeMenu}>
          NexGen<span>Forensics</span>
        </a>

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
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <path d="M6 9l6 6 6-6" />
                  </svg>
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
                            onClick={closeMenu}
                          >
                            {item.title}
                          </a>
                        </li>
                      ))}
                      <li>
                        <a className="nx-menu-more" href={group.featured.href} onClick={closeMenu}>
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
                onClick={closeMenu}
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
                  onClick={closeMenu}
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
