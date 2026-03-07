import { NextRequest, NextResponse } from "next/server";
import { exec } from "child_process";
import { promisify } from "util";
import fs from "fs";
import path from "path";
import { v4 as uuidv4 } from "uuid";

const execAsync = promisify(exec);

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

    // Paths to models and scripts
    const model1Dir = path.resolve(process.cwd(), "../model-1");
    const model2Dir = path.resolve(process.cwd(), "../model-2");
    
    // Check if model weights exist
    const model1Weights = path.join(model1Dir, "salmon_trout_binary_model.pth");
    const model2Weights = path.join(model2Dir, "mobilenet_v2_best.pth");
    
    const m1Exists = fs.existsSync(model1Weights);
    const m2Exists = fs.existsSync(model2Weights);

    // Run inference for Model 1
    let m1Result = null;
    if (m1Exists) {
        try {
            const { stdout } = await execAsync(`python3 "${path.join(model1Dir, 'inference.py')}" "${filepath}" --model_path "${model1Weights}"`);
            m1Result = JSON.parse(stdout.trim());
        } catch (error) {
            console.error("Model 1 Error:", error);
            m1Result = { error: "Inference failed" };
        }
    } else {
        m1Result = { error: "Model weights not found" };
    }

    // Run inference for Model 2
    let m2Result = null;
    if (m2Exists) {
        try {
             const { stdout } = await execAsync(`python3 "${path.join(model2Dir, 'inference.py')}" "${filepath}" --model_path "${model2Weights}"`);
             m2Result = JSON.parse(stdout.trim());
        } catch (error) {
            console.error("Model 2 Error:", error);
            m2Result = { error: "Inference failed" };
        }
    } else {
        m2Result = { error: "Model weights not found" };
    }

    // Cleanup uploaded file (optional, keeping it for now for debugging or display)
    // fs.unlinkSync(filepath);

    return NextResponse.json({
      model1: m1Result,
      model2: m2Result,
      image_url: `/uploads/${filename}`
    });

  } catch (error) {
    console.error("API Error:", error);
    return NextResponse.json({ error: "Internal Server Error" }, { status: 500 });
  }
}
