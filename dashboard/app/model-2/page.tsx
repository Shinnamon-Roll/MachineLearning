import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Hammer } from "lucide-react";

export default function Model2Page() {
  return (
    <div className="min-h-screen bg-background text-foreground p-4 md:p-8 font-sans transition-colors duration-300">
      
      {/* Header */}
      <header className="mb-10 text-center relative">
        <h1 className="text-3xl font-bold mb-2 uppercase tracking-widest border-b-4 border-foreground inline-block pb-2">
          Model 2: Pending Development
        </h1>
        <p className="text-muted-foreground mt-2">Alternative Architecture for Comparison</p>
      </header>

      <section className="flex flex-col items-center justify-center min-h-[400px] text-center space-y-6">
        <div className="bg-muted p-8 rounded-full">
            <Hammer className="w-16 h-16 text-muted-foreground" />
        </div>
        <div className="max-w-md space-y-2">
            <h2 className="text-2xl font-bold">Under Construction</h2>
            <p className="text-muted-foreground">
                This model is currently being developed. Once trained, you will be able to test it and view its performance metrics here.
            </p>
        </div>
        
        <Card className="w-full max-w-lg mt-8 bg-card border-2 border-dashed border-muted-foreground/50 shadow-none">
            <CardHeader>
                <CardTitle className="text-lg">Planned Features</CardTitle>
            </CardHeader>
            <CardContent className="text-left">
                <ul className="list-disc list-inside space-y-2 text-muted-foreground">
                    <li>Alternative Deep Learning Architecture</li>
                    <li>Comparative Analysis with Model 1</li>
                    <li>Performance Visualization</li>
                    <li>Independent Inference Testing</li>
                </ul>
            </CardContent>
        </Card>
      </section>

    </div>
  );
}
