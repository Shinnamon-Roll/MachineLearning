"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { cn } from "@/lib/utils"
import { ModeToggle } from "@/components/mode-toggle"
import { Button } from "@/components/ui/button"

export function NavBar() {
  const pathname = usePathname()

  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-14 items-center">
        <div className="mr-4 hidden md:flex">
          <Link href="/" className="mr-6 flex items-center space-x-2">
            <span className="hidden font-bold sm:inline-block">
              Salmon & Trout AI
            </span>
          </Link>
          <nav className="flex items-center gap-6 text-sm">
            <Link
              href="/"
              className={cn(
                "transition-colors hover:text-foreground/80",
                pathname === "/" ? "text-foreground font-bold" : "text-foreground/60"
              )}
            >
              Dashboard (Comparison)
            </Link>
            <Link
              href="/model-1"
              className={cn(
                "transition-colors hover:text-foreground/80",
                pathname === "/model-1" ? "text-foreground font-bold" : "text-foreground/60"
              )}
            >
              Model I
            </Link>
            <Link
              href="/model-2"
              className={cn(
                "transition-colors hover:text-foreground/80",
                pathname === "/model-2" ? "text-foreground font-bold" : "text-foreground/60"
              )}
            >
              Model II
            </Link>
          </nav>
        </div>
        <div className="flex flex-1 items-center justify-between space-x-2 md:justify-end">
          <div className="w-full flex-1 md:w-auto md:flex-none">
            {/* Search or other items can go here */}
          </div>
          <nav className="flex items-center">
            <ModeToggle />
          </nav>
        </div>
      </div>
    </header>
  )
}
