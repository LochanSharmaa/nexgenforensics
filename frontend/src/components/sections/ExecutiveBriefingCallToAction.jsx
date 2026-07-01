import { motion } from "framer-motion";
import "./ExecutiveBriefingCallToAction.css";

/**
 * Operator Access Plans Section.
 * Replaces the old Executive Briefing CTA with a highly polished pricing tier layout.
 */
export function ExecutiveBriefingCallToAction() {
  const plans = [
    {
      name: "Investigator",
      subtitle: "Individual analyst access",
      price: "$149",
      period: "/month",
      features: [
        "500 face searches/month",
        "One-to-one comparison",
        "Image authenticity checks",
        "Deepfake risk assessment",
        "PDF report generation",
        "90-day case storage",
        "HTTPS API access (2K calls)",
        "Email support",
      ],
      cta: "Start Free Trial",
      popular: false,
    },
    {
      name: "Professional",
      subtitle: "Teams & investigation units",
      price: "$499",
      period: "/month",
      features: [
        "5,000 face searches/month",
        "One-to-many batch comparison",
        "Full forensic landmark analysis",
        "Chain of custody module",
        "Multi-analyst workspaces",
        "Priority search cluster",
        "Advanced export center",
        "Audit log access",
        "Custom report templates",
        "API (50K calls/month)",
        "24/7 Priority support",
      ],
      cta: "Start Pro Trial",
      popular: true,
    },
    {
      name: "Enterprise",
      subtitle: "Agencies & enterprise deployments",
      price: "Custom",
      period: "",
      features: [
        "Unlimited searches",
        "Dedicated inference cluster",
        "On-premise deployment option",
        "Air-gapped environment",
        "Custom database integration",
        "FIPS 140-2 compliance",
        "SSO / LDAP / Active Directory",
        "SLA 99.99% uptime",
        "Dedicated forensic engineer",
        "Unlimited API access",
        "FedRAMP / CJIS ready",
      ],
      cta: "Contact Sales",
      popular: false,
    },
  ];

  return (
    <section className="nx-engage">
      <div className="nx-pricing-header">
        <p className="nx-kicker">CLEARANCE TIERS</p>
        <h2>Operator Access Plans</h2>
        <p className="nx-pricing-lead">
          Choose the tier that matches your operational requirements. All plans include chain-of-custody logging, encrypted workflows, and compliance-ready reporting.
        </p>
      </div>

      <div className="nx-pricing-grid">
        {plans.map((plan) => (
          <motion.div
            key={plan.name}
            className={`nx-pricing-card ${plan.popular ? "nx-popular" : ""}`}
            whileHover={{ y: -8, boxShadow: "0 24px 64px rgba(25, 22, 18, 0.12)" }}
            transition={{ type: "spring", stiffness: 300, damping: 20 }}
          >
            {plan.popular && <span className="nx-popular-badge">MOST POPULAR</span>}
            <div className="nx-plan-header">
              <h3>{plan.name}</h3>
              <p className="nx-plan-subtitle">{plan.subtitle}</p>
              <div className="nx-plan-price">
                <strong>{plan.price}</strong>
                {plan.period && <span>{plan.period}</span>}
              </div>
            </div>

            <ul className="nx-plan-features">
              {plan.features.map((feature, idx) => (
                <li key={idx}>
                  <svg className="nx-check-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                  <span>{feature}</span>
                </li>
              ))}
            </ul>

            <button className={`nx-plan-cta ${plan.popular ? "nx-cta-filled" : "nx-cta-outline"}`}>
              {plan.cta}
            </button>
          </motion.div>
        ))}
      </div>
    </section>
  );
}

export default ExecutiveBriefingCallToAction;
