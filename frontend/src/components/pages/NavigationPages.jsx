import { demoLinks, productLinks, resourceLinks, solutionLinks } from "../../constants/navigation";
import "./NavigationPages.css";

const routeData = [
  ...productLinks.map((item) => ({ ...item, type: "Product", groupHref: "/products" })),
  ...solutionLinks.map((item) => ({ ...item, category: "Solution", type: "Solution", groupHref: "/solutions" })),
  ...resourceLinks.map((item) => ({ ...item, category: "Resource", type: "Resource", groupHref: "/resources" })),
];

const pageCollections = {
  "/products": {
    kicker: "Product Suite",
    title: "Forensic AI Products for Recognition, Intelligence, and Case Workflows",
    body: "Explore NexGen Forensics products for biometric recognition, fingerprint examination, synthetic media analysis, OSINT, evidence graphing, and secure API integrations.",
    cards: productLinks,
  },
  "/solutions": {
    kicker: "Solutions",
    title: "Forensic AI Workflows for Modern Investigation Teams",
    body: "Choose a NexGen workflow based on your team type, evidence environment, review process, and operational requirements.",
    cards: solutionLinks,
  },
  "/resources": {
    kicker: "Resources",
    title: "Learn, Validate, and Explore Responsible Forensic AI",
    body: "Access documentation, validation notes, research material, and integration guidance for NexGen Forensics workflows.",
    cards: resourceLinks,
  },
};

export function NavigationPage({ pathname }) {
  if (pathname === "/about") return <AboutPage />;
  if (pathname === "/contact") return <ContactPage />;
  if (pathname === "/demo") return <DemoPage />;
  if (pathname.startsWith("/demo/")) return <DemoRequestPage pathname={pathname} />;

  const collection = pageCollections[pathname];
  if (collection) return <CollectionPage {...collection} />;

  const item = routeData.find((entry) => entry.href === pathname);
  if (item) return <DetailPage item={item} />;

  return <CollectionPage {...pageCollections["/products"]} />;
}

function CollectionPage({ kicker, title, body, cards }) {
  return (
    <section className="nx-page-shell" id="top">
      <PageHero kicker={kicker} title={title} body={body} />
      <div className="nx-page-card-grid">
        {cards.map((card) => (
          <a className="nx-page-card" href={card.href} key={card.href}>
            <span>{card.category || kicker}</span>
            <h3>{card.title}</h3>
            <p>{card.description}</p>
            <b>Open →</b>
          </a>
        ))}
      </div>
    </section>
  );
}

function DetailPage({ item }) {
  return (
    <section className="nx-page-shell" id="top">
      <PageHero kicker={item.type} title={item.title} body={item.description} />
      <div className="nx-detail-layout">
        <article className="nx-detail-panel">
          <span>{item.category}</span>
          <h2>Connected NexGen Workflow</h2>
          <p>
            This page is ready for the full product experience. It keeps the NexGen Forensics
            visual language while giving this workflow a real, linked destination in the site.
          </p>
          <div className="nx-page-actions">
            <a href="/demo">Try Online Demo</a>
            <a href="/contact">Request Access</a>
          </div>
        </article>
        <aside className="nx-detail-side">
          <strong>Forensic-grade interface</strong>
          <p>Secure review, expert control, audit-ready workflows, and responsible AI positioning.</p>
        </aside>
      </div>
    </section>
  );
}

function AboutPage() {
  return (
    <section className="nx-page-shell" id="top">
      <PageHero
        kicker="About Us"
        title="NexGen Forensics Builds Responsible AI Infrastructure for Investigation Teams"
        body="Our mission is to help forensic experts review evidence faster, connect intelligence clearly, and preserve expert judgment across every workflow."
      />
      <div className="nx-page-card-grid three">
        {["Expert-assisted workflows", "Responsible forensic AI", "Connected product ecosystem"].map((item) => (
          <article className="nx-page-card static" key={item}>
            <span>NexGen Value</span>
            <h3>{item}</h3>
            <p>Designed for serious teams that need clarity, auditability, and operational trust.</p>
          </article>
        ))}
      </div>
    </section>
  );
}

function ContactPage() {
  const interests = ["iMatch", "Fingerprint AI", "Deepfake Detection", "OSINT", "Evidence Graph", "API Platform"];

  return (
    <section className="nx-page-shell" id="top">
      <PageHero
        kicker="Contact"
        title="Contact NexGen Forensics"
        body="Tell us about your organization, evidence workflows, and product interest. The NexGen team will help route your request."
      />
      <div className="nx-contact-grid">
        <form className="nx-contact-form">
          {["Name", "Email", "Organization", "Role"].map((field) => (
            <label key={field}>
              <span>{field}</span>
              <input type={field === "Email" ? "email" : "text"} placeholder={field} />
            </label>
          ))}
          <label>
            <span>Product Interest</span>
            <select defaultValue="">
              <option value="" disabled>Select a product</option>
              {interests.map((interest) => (
                <option key={interest}>{interest}</option>
              ))}
            </select>
          </label>
          <label>
            <span>Message</span>
            <textarea placeholder="Share your workflow, team type, or demo request." rows="5" />
          </label>
          <button type="button">Submit Inquiry</button>
        </form>
        <div className="nx-contact-cards">
          {["Product Demo", "Partnership", "Research / Institution", "Support"].map((item) => (
            <article key={item}>
              <span>{item}</span>
              <p>Route your request to the right NexGen Forensics team.</p>
            </article>
          ))}
        </div>
      </div>
    </section>
  );
}

function DemoPage() {
  return (
    <section className="nx-page-shell" id="top">
      <PageHero
        kicker="Online Demo"
        title="Try NexGen Forensics Online Demo"
        body="Explore selected forensic AI workflows through a secure demo environment. Demo access is requested before any processing."
      />
      <div className="nx-page-card-grid">
        {demoLinks.map((demo) => (
          <a className="nx-page-card" href={demo.href} key={demo.href}>
            <span>Request Demo Access</span>
            <h3>{demo.title}</h3>
            <p>Open a polished request flow for this NexGen Forensics demo module.</p>
            <b>Request →</b>
          </a>
        ))}
      </div>
    </section>
  );
}

function DemoRequestPage({ pathname }) {
  const demo = demoLinks.find((item) => item.href === pathname) || demoLinks[0];
  return (
    <section className="nx-page-shell" id="top">
      <PageHero
        kicker="Demo Access"
        title={demo.title}
        body="Real processing is not enabled in this frontend demo. Request secure access and the NexGen team will provision the right environment."
      />
      <div className="nx-detail-layout">
        <article className="nx-detail-panel">
          <span>Secure Demo Request</span>
          <h2>Request a guided workspace</h2>
          <p>We do not fake backend results. This page collects interest and routes users to a secure demo request workflow.</p>
          <div className="nx-page-actions">
            <a href="/contact">Request Demo Access</a>
            <a href="/products">Explore Products</a>
          </div>
        </article>
      </div>
    </section>
  );
}

function PageHero({ kicker, title, body }) {
  return (
    <header className="nx-page-hero">
      <p className="nx-kicker">{kicker}</p>
      <h1>{title}</h1>
      <p>{body}</p>
    </header>
  );
}

export default NavigationPage;
