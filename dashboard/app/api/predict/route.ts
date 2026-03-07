import { NextRequest, NextResponse } from 'next/server';
import { exec } from 'child_process';
import { writeFile, unlink } from 'fs/promises';
import path from 'path';
import fs from 'fs';

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();
    const file = formData.get('file') as File;
    const modelId = formData.get('modelId') as string || 'model-2'; // Default to model-2 if not specified

    if (!file) {
      return NextResponse.json({ error: 'No file uploaded' }, { status: 400 });
    }

    const buffer = Buffer.from(await file.arrayBuffer());
    const tempDir = path.join(process.cwd(), 'public', 'temp');
    
    // Ensure temp directory exists
    if (!fs.existsSync(tempDir)) {
      fs.mkdirSync(tempDir, { recursive: true });
    }
    
    const tempFilePath = path.join(tempDir, `upload-${Date.now()}-${file.name}`);
    await writeFile(tempFilePath, buffer);

    // Determine script path based on modelId
    let scriptPath;
    if (modelId === 'model-1') {
       // Assuming model-1 has an inference script too, otherwise fallback or error
       // For now, let's use model-2 script for model-2 and handle model-1 later or use same if compatible
       // But user specifically asked about model 2.
       // Let's point model-1 to its own script if it exists, otherwise error?
       // The user said "Cleaned up Model 1 scripts: updated inference.py". So it exists.
       scriptPath = path.resolve(process.cwd(), '../model-1/inference.py');
    } else {
       scriptPath = path.resolve(process.cwd(), '../model-2/inference.py');
    }

    // Execute Python script
    // We use python3 and pass the image path
    const command = `python3 "${scriptPath}" "${tempFilePath}"`;
    
    console.log(`Executing: ${command}`);

    return new Promise((resolve) => {
      exec(command, async (error, stdout, stderr) => {
        // Clean up temp file
        try {
          await unlink(tempFilePath);
        } catch (e) {
          console.error("Failed to delete temp file:", e);
        }

        if (error) {
          console.error(`exec error: ${error}`);
          console.error(`stderr: ${stderr}`);
          resolve(NextResponse.json({ error: 'Inference failed', details: stderr }, { status: 500 }));
          return;
        }

        try {
          // Parse JSON output from python script
          // The script prints JSON to stdout
          // We need to find the JSON part in case there are other prints
          const jsonMatch = stdout.match(/\{[\s\S]*\}/);
          if (jsonMatch) {
            const result = JSON.parse(jsonMatch[0]);
            resolve(NextResponse.json(result));
          } else {
             console.error("Invalid JSON output:", stdout);
             resolve(NextResponse.json({ error: 'Invalid output from model', details: stdout }, { status: 500 }));
          }
        } catch (e) {
          console.error("JSON parse error:", e);
          resolve(NextResponse.json({ error: 'Failed to parse model output', details: stdout }, { status: 500 }));
        }
      });
    });

  } catch (error) {
    console.error('API Error:', error);
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
  }
}
