/**
 * Shrinks a data-URL image (canvas re-encode). Used so localStorage never fills up with full-size JPEGs:
 * a 640px audit frame is ~60 KB; a 240px q0.6 thumbnail is ~8 KB.
 */
export function shrinkDataUrl(
  dataUrl: string | null,
  maxWidth: number,
  quality: number
): Promise<string | null> {
  if (!dataUrl || !dataUrl.startsWith('data:image')) return Promise.resolve(dataUrl);
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const k = Math.min(1, maxWidth / (img.naturalWidth || maxWidth));
      const w = Math.max(1, Math.round((img.naturalWidth || maxWidth) * k));
      const h = Math.max(1, Math.round((img.naturalHeight || maxWidth) * k));
      const c = document.createElement('canvas');
      c.width = w;
      c.height = h;
      const ctx = c.getContext('2d');
      if (!ctx) return resolve(dataUrl);
      ctx.drawImage(img, 0, 0, w, h);
      resolve(c.toDataURL('image/jpeg', quality));
    };
    img.onerror = () => resolve(dataUrl);
    img.src = dataUrl;
  });
}
