import type { Metadata } from "next";
import "./globals.css";
import Nav from "@/components/shared/Nav";

export const metadata: Metadata = {
  title: "SIFS Creative Reasoning Engine",
  description: "AI-powered creative direction assistant for designers. One imagination. Ten reasoning lenses.",
  keywords: ["creative direction", "AI design", "concept generation", "graphic design"],
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
      </head>
      <body>
        <Nav />
        <main style={{ paddingTop: "100px" }}>{children}</main>
      </body>
    </html>
  );
}
