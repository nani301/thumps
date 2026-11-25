/**
 * Shrinks a data-URL image via offscreen canvas re-encode so localStorage never fills up
 * with full-size JPEGs: a 640px audit frame is ~60 KB; a 240px q0.60 thumbnail is ~8 KB.
 */
export function shrinkDataUrl(
  dataUrl: string | null,
  maxWidth: number,
  quality: number
): Promise<string | null> {
  if (!dataUrl || !dataUrl.startsWith('data:image')) {
    return Promise.resolve(dataUrl);
  }
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const natW = img.naturalWidth || maxWidth;
      const natH = img.naturalHeight || maxWidth;
      const k = Math.min(1, maxWidth / natW);
      const w = Math.max(1, Math.round(natW * k));
      const h = Math.max(1, Math.round(natH * k));
      const canvas = document.createElement('canvas');
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        resolve(dataUrl);
        return;
      }
      ctx.drawImage(img, 0, 0, w, h);
      try {
        resolve(canvas.toDataURL('image/jpeg', quality));
      } catch {
        resolve(dataUrl);
      }
    };
    img.onerror = () => resolve(dataUrl);
    img.src = dataUrl;
  });
}
