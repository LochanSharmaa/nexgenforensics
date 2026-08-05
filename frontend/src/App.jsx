import { Suspense, lazy } from "react";
import { BrowserRouter, Navigate, Route, Routes, useLocation } from "react-router-dom";
import { AuthProvider } from "./context/AuthContext";

import { ScrollProgressIndicator } from "./components/shared/ScrollProgressIndicator";
import { SmoothScrollProvider } from "./components/shared/SmoothScrollProvider";
import { TypographyMotionProvider } from "./components/shared/TypographyMotionProvider";
import { HeaderNavigationBar } from "./components/layout/HeaderNavigationBar";
import { FooterNavigationGrid } from "./components/layout/FooterNavigationGrid";
import { HeroForensicsIntro } from "./components/sections/HeroForensicsIntro";
import { NexFusionCorrelationSVG } from "./components/sections/NexFusionCorrelationSVG";
import { SevenDivisionsHorizontalScroll } from "./components/sections/SevenDivisionsHorizontalScroll";
import { HomeNavigationLinkSections } from "./components/sections/HomeNavigationLinkSections";
import { NexCaseCommandDashboard } from "./components/sections/NexCaseCommandDashboard";
import { RigorTrustStatements } from "./components/sections/RigorTrustStatements";
import { FutureOfForensicsQuote } from "./components/sections/FutureOfForensicsQuote";
import { ExecutiveBriefingCallToAction } from "./components/sections/ExecutiveBriefingCallToAction";
import { NavigationPage } from "./components/pages/NavigationPages";
import { ChooseRolePage } from "./components/pages/ChooseRolePage";

import "./responsive-scale.css";

/* Everything below is code-split: the landing page ships without the product
   showcases or the investigator workspace, and each chunk is fetched the
   first time its route is visited. */
const FaceSearchExperience = lazy(() =>
  import("./components/sections/FaceSearchExperience").then((m) => ({ default: m.FaceSearchExperience }))
);
const FingerprintAIPage = lazy(() =>
  import("./components/sections/FingerprintAIPage").then((m) => ({ default: m.FingerprintAIPage }))
);
const OsintProductPage = lazy(() =>
  import("./components/pages/ProductShowcasePages").then((m) => ({ default: m.OsintProductPage }))
);
const DeepfakeProductPage = lazy(() =>
  import("./components/pages/ProductShowcasePages").then((m) => ({ default: m.DeepfakeProductPage }))
);
const CrimeScene3DProductPage = lazy(() =>
  import("./components/pages/ProductShowcasePages").then((m) => ({ default: m.CrimeScene3DProductPage }))
);
const EvidenceGraphProductPage = lazy(() =>
  import("./components/pages/ProductShowcasePages").then((m) => ({ default: m.EvidenceGraphProductPage }))
);
const VideoAnalysisProductPage = lazy(() =>
  import("./components/pages/ProductShowcasePages").then((m) => ({ default: m.VideoAnalysisProductPage }))
);
const CaseIntelligenceProductPage = lazy(() =>
  import("./components/pages/ProductShowcasePages").then((m) => ({ default: m.CaseIntelligenceProductPage }))
);
const IndividualComparePage = lazy(() =>
  import("./components/pages/IndividualComparePage").then((m) => ({ default: m.IndividualComparePage }))
);
const WorkspaceLayout = lazy(() =>
  import("./workspace/WorkspaceLayout").then((m) => ({ default: m.WorkspaceLayout }))
);
const CaseListPage = lazy(() =>
  import("./workspace/CaseListPage").then((m) => ({ default: m.CaseListPage }))
);
const CaseDetailPage = lazy(() =>
  import("./workspace/CaseDetailPage").then((m) => ({ default: m.CaseDetailPage }))
);
const SearchPage = lazy(() =>
  import("./workspace/SearchPage").then((m) => ({ default: m.SearchPage }))
);
const VerifyPage = lazy(() =>
  import("./workspace/VerifyPage").then((m) => ({ default: m.VerifyPage }))
);
const BatchComparePage = lazy(() =>
  import("./workspace/BatchComparePage").then((m) => ({ default: m.BatchComparePage }))
);
const EnrolPage = lazy(() =>
  import("./workspace/EnrolPage").then((m) => ({ default: m.EnrolPage }))
);
const EnhancementPage = lazy(() =>
  import("./workspace/EnhancementPage").then((m) => ({ default: m.EnhancementPage }))
);
const AuditPage = lazy(() =>
  import("./workspace/AuditPage").then((m) => ({ default: m.AuditPage }))
);

/** Quiet full-page placeholder shown while a route chunk downloads. */
function RouteFallback() {
  return <div className="nx-route-loading" aria-busy="true" />;
}

/** The public marketing site. */
function MarketingShell({ children }) {
  return (
    <main className="nexgen-home">
      <TypographyMotionProvider />
      <SmoothScrollProvider />
      <ScrollProgressIndicator />
      <HeaderNavigationBar />
      {children}
      <FooterNavigationGrid />
    </main>
  );
}

function HomePage() {
  return (
    <>
      <HeroForensicsIntro />
      <NexFusionCorrelationSVG />
      <div id="platform">
        <SevenDivisionsHorizontalScroll />
      </div>
      <HomeNavigationLinkSections />
      <NexCaseCommandDashboard />
      <div id="validation">
        <RigorTrustStatements />
      </div>
      <FutureOfForensicsQuote />
      <div id="briefing">
        <ExecutiveBriefingCallToAction />
      </div>
    </>
  );
}

function NavigationRoute() {
  const location = useLocation();
  return <NavigationPage pathname={location.pathname} />;
}

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <Suspense fallback={<RouteFallback />}>
        <Routes>
          {/* Investigator workspace. No sign-in gate: iMATCH is an individual
              tool, and the API answers credential-less requests as the local
              owner (backend single-user mode). Role enforcement, tenancy and
              the audit trail are all still server-side. */}
          <Route path="/workspace" element={<WorkspaceLayout />}>
            <Route index element={<CaseListPage />} />
            <Route path="cases/:caseId" element={<CaseDetailPage />} />
            <Route path="search" element={<SearchPage />} />
            <Route path="enhance" element={<EnhancementPage />} />
            <Route path="verify" element={<VerifyPage />} />
            <Route path="batch-compare" element={<BatchComparePage />} />
            <Route path="enrol" element={<EnrolPage />} />
            <Route path="audit" element={<AuditPage />} />
          </Route>

          {/* The sign-in and account-lifecycle pages are retired: the old URL
              lands in the workspace it used to guard. */}
          <Route path="/login" element={<Navigate to="/workspace" replace />} />

          {/* Destination chooser. Sets a UI preference only -- it grants
              nothing and never did. */}
          <Route
            path="/choose-role"
            element={
              <MarketingShell>
                <ChooseRolePage />
              </MarketingShell>
            }
          />
          {/* The "Individual" destination: 1:1 comparison only. */}
          <Route
            path="/compare"
            element={
              <MarketingShell>
                <IndividualComparePage />
              </MarketingShell>
            }
          />
          <Route
            path="/"
            element={
              <MarketingShell>
                <HomePage />
              </MarketingShell>
            }
          />
          <Route
            path="/face-search"
            element={
              <MarketingShell>
                <FaceSearchExperience />
              </MarketingShell>
            }
          />
          <Route
            path="/products/imatch"
            element={
              <MarketingShell>
                <FaceSearchExperience />
              </MarketingShell>
            }
          />
          <Route
            path="/fingerprint-ai"
            element={
              <MarketingShell>
                <FingerprintAIPage />
              </MarketingShell>
            }
          />
          <Route
            path="/products/fingerprint-ai"
            element={
              <MarketingShell>
                <FingerprintAIPage />
              </MarketingShell>
            }
          />
          {/* Showcase pages for the home-page product deck. */}
          {[
            ["/products/osint", <OsintProductPage />],
            ["/products/deepfake-detection", <DeepfakeProductPage />],
            ["/products/3d-crime-scene", <CrimeScene3DProductPage />],
            ["/products/evidence-graph", <EvidenceGraphProductPage />],
            ["/products/video-analysis", <VideoAnalysisProductPage />],
            ["/products/case-intelligence", <CaseIntelligenceProductPage />],
          ].map(([path, element]) => (
            <Route
              key={path}
              path={path}
              element={<MarketingShell>{element}</MarketingShell>}
            />
          ))}
          {["/products/*", "/solutions/*", "/resources/*", "/demo/*", "/about", "/contact"].map((path) => (
            <Route
              key={path}
              path={path}
              element={
                <MarketingShell>
                  <NavigationRoute />
                </MarketingShell>
              }
            />
          ))}

          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
        </Suspense>
      </AuthProvider>
    </BrowserRouter>
  );
}
