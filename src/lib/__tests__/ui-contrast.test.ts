// b141.105 (audit 2026-10-23 §5.10/5.11): text tokens read ≥ 4.5:1 (WCAG AA)
// on every surface they sit on; white text only on fills that allow it.
import { describe, it, expect } from "vitest";
// No Node typings in the repo — fs through a non-literal specifier, as in
// project-contract.test.ts.
const FS_MODULE = "node:fs";
const fs: { readFileSync(u: URL, enc: string): string } = await import(/* @vite-ignore */ FS_MODULE);
const HERE = import.meta.url; // a variable: Vite rewrites new URL(literal, import.meta.url) as an asset
const css = fs.readFileSync(new URL("../../App.css", HERE), "utf8");
const root = css.slice(css.indexOf(":root {"), css.indexOf("}", css.indexOf(":root {")));

function token(name: string): string {
  const m = root.match(new RegExp(`--${name}:\\s*(#[0-9a-fA-F]{6})`));
  if (!m) throw new Error(`token --${name} not found`);
  return m[1];
}

function lum(hex: string): number {
  const c = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255)
    .map((v) => (v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4));
  return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
}
function ratio(a: string, b: string): number {
  const [x, y] = [lum(a), lum(b)].sort((p, q) => q - p);
  return (x + 0.05) / (y + 0.05);
}

const surfaces = ["bg-primary", "bg-secondary", "bg-surface", "bg-hover"];

describe("UI contrast", () => {
  for (const text of ["text-primary", "text-secondary", "text-muted"]) {
    for (const bg of surfaces) {
      it(`--${text} on --${bg} ≥ 4.5`, () => {
        expect(ratio(token(text), token(bg))).toBeGreaterThanOrEqual(4.5);
      });
    }
  }
  for (const fill of ["accent-fill", "accent-fill-hover", "danger-fill", "accent-dim"]) {
    it(`white on --${fill} ≥ 4.5`, () => {
      expect(ratio("#ffffff", token(fill))).toBeGreaterThanOrEqual(4.5);
    });
  }
  it("primary buttons use the fill tokens", () => {
    expect(css).toMatch(/\.tb-btn\.primary \{\s*background: var\(--accent-fill\)/);
    expect(css).toMatch(/\.dlg-btn-primary \{\s*background: var\(--accent-fill\)/);
  });
  it("no undeclared var(--error)", () => {
    expect(css).not.toMatch(/var\(--error\b/);
  });
});
