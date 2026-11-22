// Find the cited passage on a PDF page, so the viewer can highlight it.
//
// The snippet comes from our extracted, normalised chunk text; PDF.js gives the page as
// text items with their own spacing, hyphenation and ligatures. Comparing letters and
// digits only (lower-cased) makes the two line up. The snippet's start is searched first,
// then later anchors, so one garbled word doesn't lose the whole match.

export function normalise(s: string): string {
  return s.normalize("NFKD").toLowerCase().replace(/[^a-z0-9]/g, "");
}

/** Indices of the items that hold the snippet (empty if it isn't on the page). */
export function findHighlight(items: string[], snippet: string): number[] {
  const owner: number[] = [];
  let page = "";
  items.forEach((str, i) => {
    const n = normalise(str);
    page += n;
    for (let k = 0; k < n.length; k++) owner.push(i);
  });
  const needle = normalise(snippet);
  if (!needle || !page) return [];

  if (needle.length <= 24) {
    const at = page.indexOf(needle);
    return at === -1 ? [] : [...new Set(owner.slice(at, at + needle.length))];
  }
  for (const len of [60, 40, 24]) {
    for (const frac of [0, 0.33, 0.66]) {
      const offset = Math.floor((needle.length - len) * frac);
      if (offset < 0) continue;
      const anchor = needle.slice(offset, offset + len);
      if (anchor.length < Math.min(len, needle.length)) continue;
      const at = page.indexOf(anchor);
      if (at === -1) continue;
      const start = Math.max(0, at - offset);
      const end = Math.min(page.length, start + needle.length);
      return [...new Set(owner.slice(start, end))];
    }
  }
  return [];
}
