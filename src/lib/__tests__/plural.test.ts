import { describe, it, expect } from "vitest";
import { pluralRu } from "../plural";

describe("pluralRu", () => {
  const f = (n: number) => `${n} ${pluralRu(n, "бэнд", "бэнда", "бэндов")}`;
  it("covers 1 / 2–4 / 5+ and the 11–14 exception", () => {
    expect([1, 2, 4, 5, 8, 11, 12, 14, 21, 22, 25, 111].map(f)).toEqual([
      "1 бэнд", "2 бэнда", "4 бэнда", "5 бэндов", "8 бэндов", "11 бэндов",
      "12 бэндов", "14 бэндов", "21 бэнд", "22 бэнда", "25 бэндов", "111 бэндов",
    ]);
  });
});
