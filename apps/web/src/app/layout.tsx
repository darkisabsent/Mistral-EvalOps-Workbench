import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import NavSidebar from "../../components/NavSidebar";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Mistral EvalOps Workbench",
  description: "RAG chat, A/B experiments, eval runs, and observability.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-neutral-950 text-neutral-100`}
      >
        <div className="flex h-screen">
          <NavSidebar />
          <main className="flex-1 overflow-y-auto px-8 py-6 space-y-6">
            {children}
          </main>
        </div>
      </body>
    </html>
  );
}
