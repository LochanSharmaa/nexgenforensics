import { useMemo } from "react";
import "./FooterNavigationGrid.css";

/**
 * Site footer displaying branding info, category navigation links,
 * and standard dynamic copyright year indicator.
 */
export function FooterNavigationGrid() {
  const year = useMemo(() => new Date().getFullYear(), []);

  return (
    <footer className="nx-footer">
      <div>
        <strong>NexGen Forensics</strong>
        <p>Building the operational infrastructure for the future of forensic intelligence.</p>
      </div>
      <div className="nx-footer-grid">
        <a href="/products/imatch">iMatch</a>
        <a href="/products/fingerprint-ai">Fingerprint AI</a>
        <a href="/products/deepfake-detection">Deepfake Detection</a>
        <a href="/products/osint">OSINT</a>
        <a href="/products/evidence-graph">Evidence Graph</a>
        <a href="/products/case-intelligence">Case Intelligence</a>
        <a href="/resources/api-reference">API Reference</a>
        <a href="/resources/validation">Validation</a>
      </div>
      <div className="nx-footer-legal">
        <small>© {year} NexGen Forensics</small>
      </div>
    </footer>
  );
}

export default FooterNavigationGrid;
