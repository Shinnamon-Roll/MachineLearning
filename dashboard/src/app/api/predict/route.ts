import { NextRequest, NextResponse } from "next/server";
import { execFile } from "child_process";
import { promisify } from "util";
import fs from "fs";
import path from "path";
import { v4 as uuidv4 } from "uuid";

const execFileAsync = promisify(execFile);

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();
    const file = formData.get("file") as File;

    if (!file) {
      return NextResponse.json({ error: "No file uploaded" }, { status: 400 });
    }

    // Create uploads directory if it doesn't exist
    const uploadsDir = path.join(process.cwd(), "public", "uploads");
    if (!fs.existsSync(uploadsDir)) {
      fs.mkdirSync(uploadsDir, { recursive: true });
    }

    // Save file
    const buffer = Buffer.from(await file.arrayBuffer());
    const filename = `${uuidv4()}${path.extname(file.name)}`;
    const filepath = path.join(uploadsDir, filename);
    fs.writeFileSync(filepath, buffer);

    // Both models are served by the same script; only --model differs
    const mlDir = path.resolve(process.cwd(), "../ml");
    const inferenceScript = path.join(mlDir, "inference.py");

    const runModel = async (name: "densenet" | "mobilenet") => {
      const weights = path.join(mlDir, "weights", `${name}.pth`);
      if (!fs.existsSync(weights)) {
        return { error: "Model weights not found" };
      }
      try {
        // execFile passes args without a shell, so the uploaded filename cannot inject commands
        const { stdout } = await execFileAsync("python3", [inferenceScript, filepath, "--model", name]);
        return JSON.parse(stdout.trim());
      } catch (error) {
        console.error(`${name} Error:`, error);
        return { error: "Inference failed" };
      }
    };

    const [m1Result, m2Result] = await Promise.all([runModel("densenet"), runModel("mobilenet")]);

    // Thai text summary of both results; a failure here must not hide the predictions
    let explanation = null;
    try {
      const { stdout, stderr } = await execFileAsync("python3", [
        path.join(mlDir, "explain.py"),
        filepath,
        "--predictions",
        JSON.stringify({ densenet: m1Result, mobilenet: m2Result }),
      ]);
      if (stderr.trim()) console.warn("Explain:", stderr.trim());
      explanation = JSON.parse(stdout.trim());
    } catch (error) {
      console.error("Explain Error:", error);
    }

    // Cleanup uploaded file (optional, keeping it for now for debugging or display)
    // fs.unlinkSync(filepath);

    return NextResponse.json({
      model1: m1Result,
      model2: m2Result,
      explanation,
      image_url: `/uploads/${filename}`
    });

  } catch (error) {
    console.error("API Error:", error);
    return NextResponse.json({ error: "Internal Server Error" }, { status: 500 });
  }
}
