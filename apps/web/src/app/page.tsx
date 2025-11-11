import Link from "next/link";

export default function Home() {
  return (
    <main className="container mx-auto max-w-screen-xl px-4 md:px-6 space-y-6">
      <h1 className="text-3xl font-bold">Mistral EvalOps Workbench</h1>
      <p className="text-neutral-300">Grounded chat, versioned prompts, offline judge evaluations, and full observability.</p>
      <div className="grid auto-rows-fr grid-cols-[repeat(auto-fit,minmax(12rem,1fr))] gap-4 md:gap-6">
        {[
          { href:"/chat", label:"Chat", desc:"Stream grounded answers with citations." },
          { href:"/prompts", label:"Prompts", desc:"Version templates & run A/B." },
          { href:"/datasets", label:"Datasets", desc:"Upload QA sets & preview." },
          { href:"/runs", label:"Runs", desc:"Inspect latency, tokens, cost." },
        ].map(c=>(
          <Link
            key={c.href}
            href={c.href}
            className="group h-full rounded border border-neutral-800 bg-neutral-900 p-4 hover:border-indigo-600 transition flex flex-col justify-between"
          >
            <div>
              <div className="text-sm font-semibold mb-1">{c.label}</div>
              <div className="text-xs text-neutral-400">{c.desc}</div>
            </div>
            <div className="text-indigo-400 text-xs mt-3 group-hover:underline">Open →</div>
          </Link>
        ))}
      </div>
    </main>
  );
}
