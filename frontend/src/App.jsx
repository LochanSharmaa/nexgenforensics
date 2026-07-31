import { BrowserRouter, Navigate, Route, Routes, useLocation } from "react-router-dom";
import { AuthProvider, useAuth } from "./context/AuthContext";

import { ScrollProgressIndicator } from "./components/shared/ScrollProgressIndicator";
import { SmoothScrollProvider } from "./components/shared/SmoothScrollProvider";
import { TypographyMotionProvider } from "./components/shared/TypographyMotionProvider";
import { HeaderNavigationBar } from "./components/layout/HeaderNavigationBar";
import { FooterNavigationGrid } from "./components/layout/FooterNavigationGrid";
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
import { NavigationPage } from "./components/pages/NavigationPages";
import { LoginPage } from "./components/pages/LoginPage";
import { ChooseRolePage } from "./components/pages/ChooseRolePage";
import {
  ForgotPasswordPage,
  RegisterPage,
  ResetPasswordPage,
  VerifyEmailPage,
} from "./components/pages/AuthFlowPages";
import { IndividualComparePage } from "./components/pages/IndividualComparePage";

import { WorkspaceLayout } from "./workspace/WorkspaceLayout";
import { CaseListPage } from "./workspace/CaseListPage";
import { CaseDetailPage } from "./workspace/CaseDetailPage";
import { SearchPage } from "./workspace/SearchPage";
import { VerifyPage } from "./workspace/VerifyPage";
import { EnrolPage } from "./workspace/EnrolPage";
import { AuditPage } from "./workspace/AuditPage";

import "./responsive-scale.css";

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
            <Route path="verify" element={<VerifyPage />} />
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
      </AuthProvider>
    </BrowserRouter>
  );
}
