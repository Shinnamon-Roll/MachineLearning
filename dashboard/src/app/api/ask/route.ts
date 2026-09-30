import { NextRequest, NextResponse } from "next/server";
import fs from "fs";
import path from "path";

// Q&A about the latest prediction. Gemini answers from the same grounded facts the
// explanation was built from, plus the project's test-set metrics.
const MODELS = [process.env.GEMINI_MODEL || "gemini-flash-latest", "gemini-flash-lite-latest"];
const MAX_QUESTION = 500;
const MAX_TURNS = 6;

type Turn = { role: "user" | "model"; text: string };

function projectFacts() {
  const dir = path.join(process.cwd(), "public", "data");
  return ["metrics.json", "metrics_model2.json"]
    .map((f) => {
      try {
        const m = JSON.parse(fs.readFileSync(path.join(dir, f), "utf8"));
        const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
        const recall = Object.entries(m.recall_per_class ?? {})
          .map(([c, v]) => `${c} ${pct(v as number)}`)
          .join(", ");
        return `${m.model_name}: test accuracy ${pct(m.accuracy)}, F1 ${pct(m.f1_score)}, recall ต่อคลาส ${recall}, ` +
          `params ${m.params}, ขนาด ${m.size}, latency ${m.inference}/ภาพ`;
      } catch {
        return null;
      }
    })
    .filter(Boolean)
    .join("\n");
}

const SYSTEM = (facts: string) => `คุณคือผู้ช่วยตอบคำถามเกี่ยวกับผลการจำแนกภาพเนื้อปลา Salmon กับ Trout บนเว็บนี้ ตอบเป็นภาษาไทย กระชับ 1-4 ประโยค ไม่ใช้ markdown
กฎ:
- ตอบจากข้อมูลด้านล่างเป็นหลัก ห้ามแต่งตัวเลขหรือผลที่ไม่มีในข้อมูล
- ห้ามเปรียบเทียบตัวเลขเอง ให้ใช้คำบรรยายตามข้อมูล
- ถ้าคำถามต้องใช้ความรู้ทั่วไปนอกข้อมูล (เช่นเรื่องปลา) ตอบสั้น ๆ และบอกว่าเป็นความรู้ทั่วไป ไม่ได้มาจากโมเดล
- ถ้าไม่เกี่ยวกับปลาหรือโปรเจ็คนี้ ให้บอกสุภาพว่าตอบได้เฉพาะเรื่องนี้
- เขียนชื่อปลาเป็น Salmon และ Trout
- คำอธิบายสีเนื้อปลาเทียบกับค่าเฉลี่ยจากภาพที่ใช้ฝึก ไม่ใช่สิ่งที่ CNN มองจริง ถ้าถูกถามว่าโมเดลดูอะไร ให้บอกข้อจำกัดนี้

ผลของภาพล่าสุด:
${facts || "ยังไม่มีภาพ"}

ผลของโมเดลบน test set (140 ภาพ):
${projectFacts()}

ความรู้เกี่ยวกับโปรเจ็ค: เทียบ DenseNet121 กับ MobileNetV2 ด้วยสูตรเทรนเดียวกัน (transfer learning 2 phase, Focal Loss, CLAHE, 3 seeds) คำอธิบายสร้างจาก template ที่เทียบความสว่าง ความแดง ความสดของสีเนื้อปลากับค่าเฉลี่ยของแต่ละคลาส แล้วให้ Gemini เรียบเรียง`;

export async function POST(req: NextRequest) {
  const key = process.env.GEMINI_API_KEY;
  if (!key) return NextResponse.json({ error: "ยังไม่ได้ตั้งค่า GEMINI_API_KEY" }, { status: 503 });

  let body: { question?: unknown; facts?: unknown; history?: unknown };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
  }
  const question = typeof body.question === "string" ? body.question.trim() : "";
  if (!question || question.length > MAX_QUESTION) {
    return NextResponse.json({ error: `คำถามต้องยาว 1-${MAX_QUESTION} ตัวอักษร` }, { status: 400 });
  }
  const facts = typeof body.facts === "string" ? body.facts.slice(0, 4000) : "";
  const history = (Array.isArray(body.history) ? body.history : [])
    .filter((t): t is Turn => (t?.role === "user" || t?.role === "model") && typeof t?.text === "string")
    .slice(-MAX_TURNS);

  const payload = JSON.stringify({
    systemInstruction: { parts: [{ text: SYSTEM(facts) }] },
    contents: [...history, { role: "user", text: question }].map((t) => ({
      role: t.role,
      parts: [{ text: t.text.slice(0, 2000) }],
    })),
    generationConfig: { temperature: 0.2 },
  });

  let lastError = "";
  // Free tier often answers 503 "overloaded": retry once, then try the lighter model
  for (const model of [...new Set(MODELS)]) {
    for (let attempt = 0; attempt < 2; attempt++) {
      try {
        const res = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`, {
          method: "POST",
          headers: { "Content-Type": "application/json", "x-goog-api-key": key },
          body: payload,
          signal: AbortSignal.timeout(20000),
        });
        if (res.ok) {
          const data = await res.json();
          const answer = data?.candidates?.[0]?.content?.parts?.map((p: { text?: string }) => p.text ?? "").join("").trim();
          if (answer) return NextResponse.json({ answer });
          lastError = `${model}: empty answer`;
          break;
        }
        lastError = `${model}: HTTP ${res.status}`;
        if (![429, 500, 503].includes(res.status)) break;
        await new Promise((r) => setTimeout(r, 1000 * (attempt + 1)));
      } catch (e) {
        lastError = `${model}: ${e}`;
        break;
      }
    }
  }
  console.error("Ask Error:", lastError);
  return NextResponse.json({ error: "Gemini ไม่ว่างตอนนี้ ลองใหม่อีกครั้ง" }, { status: 502 });
}
