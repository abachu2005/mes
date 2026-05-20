import { Routes, Route, Navigate, Link, useLocation } from "react-router-dom";
import { Activity, Moon, Sun, BookOpen, Users, FlaskConical, GitBranch } from "lucide-react";
import { Landing } from "./pages/Landing";
import { Dashboard } from "./pages/Dashboard";
import { ParticipantDetail } from "./pages/ParticipantDetail";
import { SessionUpload } from "./pages/SessionUpload";
import { SessionReport } from "./pages/SessionReport";
import { Comparison } from "./pages/Comparison";
import { About } from "./pages/About";
import { useTheme } from "./lib/theme";
import { cn } from "./lib/cn";

export function App() {
  const { dark, toggle } = useTheme();
  const loc = useLocation();
  const isLanding = loc.pathname === "/";

  return (
    <div className="min-h-screen flex flex-col">
      {!isLanding && (
        <header className="bg-white dark:bg-ink-900 border-b border-ink-200 dark:border-ink-800 sticky top-0 z-30">
          <div className="max-w-7xl mx-auto px-6 h-14 flex items-center justify-between">
            <Link to="/" className="flex items-center gap-2 font-semibold">
              <span className="grid place-items-center w-8 h-8 rounded-lg bg-teal-600 text-white">
                <Activity className="w-4 h-4" />
              </span>
              <span>MES</span>
              <span className="text-ink-400 text-sm hidden sm:inline">— Motor Engagement Signal</span>
            </Link>
            <nav className="flex items-center gap-1">
              <NavLink to="/dashboard" icon={<Users className="w-4 h-4" />}>Participants</NavLink>
              <NavLink to="/compare" icon={<GitBranch className="w-4 h-4" />}>Compare</NavLink>
              <NavLink to="/about" icon={<BookOpen className="w-4 h-4" />}>About</NavLink>
              <button onClick={toggle} className="btn-ghost" aria-label="toggle dark mode">
                {dark ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
              </button>
            </nav>
          </div>
        </header>
      )}

      <main className="flex-1 animate-fade-in">
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/participants/:id" element={<ParticipantDetail />} />
          <Route path="/participants/:id/upload" element={<SessionUpload />} />
          <Route path="/sessions/:id" element={<SessionReport />} />
          <Route path="/compare" element={<Comparison />} />
          <Route path="/about" element={<About />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>

      <footer className={cn("py-4 px-6 text-xs text-ink-500 dark:text-ink-400 text-center", isLanding && "border-t border-ink-200 dark:border-ink-800")}>
        <span className="inline-flex items-center gap-2">
          <FlaskConical className="w-3.5 h-3.5" />
          Research use only · Not FDA / CE cleared
        </span>
        <span className="mx-2 text-ink-300">·</span>
        <a href="https://huggingface.co/spaces/abachu2005/mes" className="hover:underline">Hugging Face Space</a>
        <span className="mx-2 text-ink-300">·</span>
        <span>"Quantifying neural drive for movement recovery."</span>
      </footer>
    </div>
  );
}

function NavLink({ to, icon, children }: { to: string; icon: React.ReactNode; children: React.ReactNode }) {
  const loc = useLocation();
  const active = loc.pathname === to || loc.pathname.startsWith(to + "/");
  return (
    <Link
      to={to}
      className={cn(
        "btn-ghost text-sm px-3 py-1.5",
        active && "bg-ink-100 dark:bg-ink-800 text-teal-700 dark:text-teal-300",
      )}
    >
      {icon}
      <span className="hidden sm:inline">{children}</span>
    </Link>
  );
}
