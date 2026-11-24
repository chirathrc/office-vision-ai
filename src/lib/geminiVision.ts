/**
 * Optional Gemini Vision deep scene analyzer for nuanced office activities
 * (e.g., meeting discussions, sleeping/resting, typing vs browsing, phone calls).
 */
export async function analyzeFrameWithGemini(
  apiKey: string,
  base64Jpeg: string
): Promise<string> {
  const cleanBase64 = base64Jpeg.replace(/^data:image\/\w+;base64,/, "");
  const endpoint = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${encodeURIComponent(
    apiKey.trim()
  )}`;

  const prompt = `You are an Office Activity Vision AI assistant. Analyze this office camera frame concisely:
1. Count how many people are visible.
2. For each person (ordered left to right), state:
   - Person #N: Activity (Working at PC / On Phone / In Meeting / Reading / On Break / Idle)
   - Brief 1-line observation of what they are doing.
3. Give a 1-line overall office summary.
Keep it concise, objective, and bulleted.`;

  const res = await fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      contents: [
        {
          parts: [
            { text: prompt },
            {
              inline_data: {
                mime_type: "image/jpeg",
                data: cleanBase64,
              },
            },
          ],
        },
      ],
    }),
  });

  if (!res.ok) {
    const errText = await res.text();
    throw new Error(`Gemini API error (${res.status}): ${errText}`);
  }

  const json = await res.json();
  return (
    json?.candidates?.[0]?.content?.parts?.[0]?.text ||
    "No analysis text returned."
  );
}
