import type { Metadata, Viewport } from "next";
import { Manrope, Saira_Stencil } from "next/font/google";
import "./globals.css";
import SmoothScroll from "@/components/SmoothScroll";

const manrope = Manrope({ subsets: ["latin"], variable: "--font-manrope", weight: ["400", "500", "600", "700", "800"] });
const stencil = Saira_Stencil({ subsets: ["latin"], variable: "--font-saira-stencil", weight: "700" });

export const metadata: Metadata = {
  title: "Smoke Rescue",
  description: "Robot dogs map a burning house, find the people inside, and tell the crew where to go.",
};

export const viewport: Viewport = { themeColor: "#0b0612", viewportFit: "cover" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${manrope.variable} ${stencil.variable}`}>
      <body>
        <SmoothScroll>{children}</SmoothScroll>
      </body>
    </html>
  );
}
