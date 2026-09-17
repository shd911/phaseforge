/** Russian plural form: ("бэнд", "бэнда", "бэндов") for 1 / 2–4 / 5+ (and 11–14). */
export function pluralRu(n: number, one: string, few: string, many: string): string {
  const a = Math.abs(n) % 100;
  const b = a % 10;
  if (a >= 11 && a <= 14) return many;
  if (b === 1) return one;
  if (b >= 2 && b <= 4) return few;
  return many;
}
