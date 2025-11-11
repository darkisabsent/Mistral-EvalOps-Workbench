"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";

const items = [
  { href: "/", label: "Home" },
  { href: "/chat", label: "Chat" },
  { href: "/prompts", label: "Prompts" },
  { href: "/datasets", label: "Datasets" },
  { href: "/runs", label: "Runs" },
];

export default function NavSidebar() {
  const pathname = usePathname();
  return (
    <aside className="w-60 bg-neutral-900 border-r border-neutral-800 flex flex-col">
      <div className="px-5 py-4 border-b border-neutral-800">
        <h1 className="text-lg font-semibold tracking-tight">EvalOps</h1>
        <p className="text-xs text-neutral-400 mt-1">Grounded RAG + A/B + Judge</p>
      </div>
      <nav className="flex-1 overflow-y-auto py-4">
        <ul className="space-y-1 px-3">
          {items.map(i => {
            const active = pathname === i.href;
            return (
              <li key={i.href}>
                <Link
                  href={i.href}
                  className={`block rounded px-3 py-2 text-sm transition ${
                    active
                      ? "bg-indigo-600 text-white shadow"
                      : "text-neutral-300 hover:bg-neutral-800 hover:text-white"
                  }`}
                >
                  {i.label}
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>
    </aside>
  );
}
