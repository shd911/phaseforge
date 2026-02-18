/**
 * Floor Bounce Analysis
 *
 * Вычисляет частоты гребенчатой фильтрации при отражении звука от пола
 * между динамиком и микрофоном.
 *
 * Геометрия:
 *   - Прямой путь: расстояние от динамика до микрофона (D)
 *   - Отражённый путь: sqrt(D^2 + (Hs + Hm)^2)
 *     где Hs — высота динамика, Hm — высота микрофона над полом
 *   - Разница путей: отражённый - прямой
 *   - Первая деструктивная интерференция (null):
 *       f_null = c / (2 * path_difference)
 *   - Nulls: f_null * 1, 3, 5, 7, ...
 *   - Peaks:  f_null * 2, 4, 6, 8, ...
 */

const SPEED_OF_SOUND = 343; // м/с

export interface FloorBounceResult {
  pathDifference: number; // метры
  firstNullFreq: number; // Гц
  nullFreqs: number[]; // частоты деструктивных интерференций
  peakFreqs: number[]; // частоты конструктивных интерференций
}

/**
 * Вычислить частоты гребенчатой фильтрации floor bounce
 *
 * @param speakerHeight — высота динамика над полом (м)
 * @param micHeight — высота микрофона над полом (м)
 * @param distance — горизонтальное расстояние динамик-микрофон (м)
 * @param maxFreq — максимальная частота для вычисления (по умолчанию 20000 Гц)
 */
export function computeFloorBounce(
  speakerHeight: number,
  micHeight: number,
  distance: number,
  maxFreq: number = 20000,
): FloorBounceResult {
  // Прямой путь (горизонтальное расстояние)
  const directPath = Math.sqrt(
    distance * distance +
    (speakerHeight - micHeight) * (speakerHeight - micHeight)
  );

  // Отражённый путь (через виртуальный источник зеркальное отражение)
  const reflectedPath = Math.sqrt(
    distance * distance +
    (speakerHeight + micHeight) * (speakerHeight + micHeight)
  );

  const pathDifference = reflectedPath - directPath;

  if (pathDifference <= 0) {
    return { pathDifference: 0, firstNullFreq: 0, nullFreqs: [], peakFreqs: [] };
  }

  const firstNullFreq = SPEED_OF_SOUND / (2 * pathDifference);

  const nullFreqs: number[] = [];
  const peakFreqs: number[] = [];

  // Nulls: f_null * (2n-1) для n=1,2,3,...
  for (let n = 1; ; n++) {
    const f = firstNullFreq * (2 * n - 1);
    if (f > maxFreq) break;
    nullFreqs.push(f);
  }

  // Peaks: f_null * 2n для n=1,2,3,...
  for (let n = 1; ; n++) {
    const f = firstNullFreq * 2 * n;
    if (f > maxFreq) break;
    peakFreqs.push(f);
  }

  return { pathDifference, firstNullFreq, nullFreqs, peakFreqs };
}
