import { ScrollProgressIndicator } from "./components/shared/ScrollProgressIndicator";
import { SmoothScrollProvider } from "./components/shared/SmoothScrollProvider";
import { TypographyMotionProvider } from "./components/shared/TypographyMotionProvider";
import { HeaderNavigationBar } from "./components/layout/HeaderNavigationBar";
import { HeroForensicsIntro } from "./components/sections/HeroForensicsIntro";
import { EvidenceScatteringScroll } from "./components/sections/EvidenceScatteringScroll";
import { NexFusionCorrelationSVG } from "./components/sections/NexFusionCorrelationSVG";
import { SevenDivisionsHorizontalScroll } from "./components/sections/SevenDivisionsHorizontalScroll";
import { HomeNavigationLinkSections } from "./components/sections/HomeNavigationLinkSections";
import { EnterprisePlatformOverview } from "./components/sections/EnterprisePlatformOverview";
import { NexCaseCommandDashboard } from "./components/sections/NexCaseCommandDashboard";
import { RigorTrustStatements } from "./components/sections/RigorTrustStatements";
import { InstitutionalResearchGrid } from "./components/sections/InstitutionalResearchGrid";
import { FutureOfForensicsQuote } from "./components/sections/FutureOfForensicsQuote";
import { ExecutiveBriefingCallToAction } from "./components/sections/ExecutiveBriefingCallToAction";
import { FaceSearchExperience } from "./components/sections/FaceSearchExperience";
import { FingerprintAIPage } from "./components/sections/FingerprintAIPage";
import { LoginPage } from "./components/pages/LoginPage";
import { NavigationPage } from "./components/pages/NavigationPages";
import { FooterNavigationGrid } from "./components/layout/FooterNavigationGrid";
import "./responsive-scale.css";

export default function App() {
  const pathname = window.location.pathname;
  const isLoginPage = pathname === "/login";
  const isFaceSearchPage = pathname === "/face-search" || pathname === "/products/imatch";
  const isFingerprintPage = pathname === "/fingerprint-ai" || pathname === "/products/fingerprint-ai";
  const isNavigationPage =
    pathname.startsWith("/products") ||
    pathname.startsWith("/solutions") ||
    pathname.startsWith("/resources") ||
    pathname.startsWith("/demo") ||
    pathname === "/about" ||
    pathname === "/contact";

  return (
    <main className="nexgen-home">
      <TypographyMotionProvider />
      <SmoothScrollProvider />
      <ScrollProgressIndicator />
      <HeaderNavigationBar />

      {isLoginPage ? (
        <LoginPage />
      ) : isFingerprintPage ? (
        <FingerprintAIPage />
      ) : isFaceSearchPage ? (
        <FaceSearchExperience />
      ) : isNavigationPage ? (
        <NavigationPage pathname={pathname} />
      ) : (
        <>
          <HeroForensicsIntro />
          <EvidenceScatteringScroll />
          <NexFusionCorrelationSVG />

          <div id="platform">
            <SevenDivisionsHorizontalScroll />
          </div>

          <HomeNavigationLinkSections />

          <EnterprisePlatformOverview />

          <NexCaseCommandDashboard />

          <div id="validation">
            <RigorTrustStatements />
          </div>

          <div id="research">
            <InstitutionalResearchGrid />
          </div>

          <FutureOfForensicsQuote />

          <div id="briefing">
            <ExecutiveBriefingCallToAction />
          </div>
        </>
      )}

      <FooterNavigationGrid />
    </main>
  );
}
