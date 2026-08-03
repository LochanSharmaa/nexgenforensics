import { Suspense, lazy } from "react";
import { BrowserRouter, Navigate, Route, Routes, useLocation } from "react-router-dom";
import { AuthProvider, useAuth } from "./context/AuthContext";

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
import { LoginPage } from "./components/pages/LoginPage";
import { ChooseRolePage } from "./components/pages/ChooseRolePage";

import "./responsive-scale.css";

/* Everything below is code-split: the landing page ships without the product
   showcases, auth flows, or the investigator workspace, and each chunk is
   fetched the first time its route is visited. */
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
const RegisterPage = lazy(() =>
  import("./components/pages/AuthFlowPages").then((m) => ({ default: m.RegisterPage }))
);
const VerifyEmailPage = lazy(() =>
  import("./components/pages/AuthFlowPages").then((m) => ({ default: m.VerifyEmailPage }))
);
const ForgotPasswordPage = lazy(() =>
  import("./components/pages/AuthFlowPages").then((m) => ({ default: m.ForgotPasswordPage }))
);
const ResetPasswordPage = lazy(() =>
  import("./components/pages/AuthFlowPages").then((m) => ({ default: m.ResetPasswordPage }))
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

/**
 * Gate for the workspace.
 *
 * This is a usability boundary, not a security boundary: it decides what to
 * render, and nothing more. Authorization is enforced by the API on every
 * request, so removing this component would make the UI confusing but would not
 * expose a single record.
 */
function RequireAuth({ children, minRole }) {
  const { isAuthenticated, loading, hasRole } = useAuth();
  const location = useLocation();

  if (loading) {
    return <div className="wk-loading" style={{ padding: 80 }}>Restoring session…</div>;
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace state={{ from: location }} />;
  }

  if (minRole && !hasRole(minRole)) {
    return (
      <div className="wk-main">
        <div className="wk-banner critical">
          <div>
            <strong>Insufficient permissions</strong>
            This area requires the {minRole} role. Ask an administrator in your organisation if
            you need access.
          </div>
        </div>
      </div>
    );
  }

  return children;
}

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <Suspense fallback={<RouteFallback />}>
        <Routes>
          {/* Investigator workspace */}
          <Route
            path="/workspace"
            element={
              <RequireAuth>
                <WorkspaceLayout />
              </RequireAuth>
            }
          >
            <Route index element={<CaseListPage />} />
            <Route path="cases/:caseId" element={<CaseDetailPage />} />
            <Route path="search" element={<SearchPage />} />
            <Route path="enhance" element={<EnhancementPage />} />
            <Route path="verify" element={<VerifyPage />} />
            <Route path="batch-compare" element={<BatchComparePage />} />
            <Route
              path="enrol"
              element={
                <RequireAuth minRole="supervisor">
                  <EnrolPage />
                </RequireAuth>
              }
            />
            <Route path="audit" element={<AuditPage />} />
          </Route>

          {/* Public site */}
          <Route
            path="/login"
            element={
              <MarketingShell>
                <LoginPage />
              </MarketingShell>
            }
          />
          {/* Account lifecycle: all public, all reachable without a session --
              they exist for people who cannot sign in yet. */}
          {[
            ["/register", <RegisterPage />],
            ["/verify-email", <VerifyEmailPage />],
            ["/forgot-password", <ForgotPasswordPage />],
            ["/reset-password", <ResetPasswordPage />],
          ].map(([path, element]) => (
            <Route
              key={path}
              path={path}
              element={<MarketingShell>{element}</MarketingShell>}
            />
          ))}

          {/* Post-login destination chooser. Sets a UI preference only -- it
              grants nothing, so it sits behind the same auth gate as anything
              else that assumes a signed-in user. */}
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
              <RequireAuth>
                <MarketingShell>
                  <IndividualComparePage />
                </MarketingShell>
              </RequireAuth>
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
