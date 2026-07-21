import handImage from "../../assets/hand-hd.png";
import "./HomeNavigationLinkSections.css";

export function HomeNavigationLinkSections() {
  return (
    <section className="nx-home-fingerprint-slide" style={{ "--hand-bg": `url(${handImage})` }}>
      <h2>Automated Forensic Examination</h2>
    </section>
  );
}

export default HomeNavigationLinkSections;
