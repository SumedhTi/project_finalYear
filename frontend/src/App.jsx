import { useState } from "react";
import { useRef } from "react";

export default function App() {
  const BASE = "https://1b24-103-106-200-58.ngrok-free.app";
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [output, setOutput] = useState(null);
  const [loading, setLoading] = useState(false);
  const [selectedTool, setSelectedTool] = useState(null);
  const [compression, setCompression] = useState(50);
  const [lowlightIntensity, setLowlightIntensity] = useState(3.0);
  const fileInputRef = useRef(null);

  const tools = ["Deblur", "Compress", "Text Detect", "Denoise", "Brighten"];

  const triggerUpload = () => {
    fileInputRef.current.click();
  };

  const handleUpload = (e) => {
    const f = e.target.files[0];
    if (!f) return;

    setFile(f);
    setPreview(URL.createObjectURL(f));
    setOutput(null);
    setSelectedTool(null);
  };

  const runTool = async (tool) => {
    // Special behavior for compression and brighten
    if (tool === "Compress") {
      // If slider not opened yet → just open it
      if (selectedTool !== "Compress") {
        setSelectedTool("Compress");
        return;
      }
    }
    if (tool === "Brighten") {
      // If slider not opened yet → just open it
      if (selectedTool !== "Brighten") {
        setSelectedTool("Brighten");
        return;
      }
    }

    setSelectedTool(tool);
    setLoading(true);

    const formData = new FormData();
    formData.append("file", file);

    let endpoint = "";
    let method = "POST";
    let body = formData;

    try {
      const isVideo = file.type.startsWith("video");

      if (tool === "Text Detect") {
        endpoint = BASE + "/ocr";
      } else if (tool === "Compress") {
        endpoint = isVideo
          ? `${BASE}/process-video?mode=Compress&quality=${compression}`
          : `${BASE}/compress?quality=${compression}`;
      } else if (tool === "Brighten") {
        endpoint = isVideo
          ? `${BASE}/process-video?mode=LowLight&intensity=${lowlightIntensity}`
          : `${BASE}/lowlight?intensity=${lowlightIntensity}`;
      } else {
        const mode = tool; // Deblur or Denoise
        endpoint = isVideo
          ? `${BASE}/process-video?mode=${mode}`
          : `${BASE}/process-image?mode=${mode}`;
      }

      const res = await fetch(endpoint, {
        method: method,
        body: body,
      });

      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }

      if (tool === "Text Detect") {
        const data = await res.json();
        setOutput(data.text);
      } else {
        const data = await res.json();
        const url = data.image_url || data.video_url || "";
        setOutput(url);
      }
    } catch (err) {
      console.error("Processing error:", err);
      setOutput("Error processing file. Please try again.");
    }

    setLoading(false);
  };

  const downloadFile = () => {
    if (!output) return;

    // Create a temporary link element
    const a = document.createElement("a");
    a.href = output; // This is already a data URL from the backend
    a.download = `processed_${file.name}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white flex flex-col items-center p-6">
      <h1 className="text-3xl font-bold mb-6">Media Processing Tools</h1>

      {/* Upload */}
      <div className="mb-8 flex flex-col items-center">
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*,video/*"
          onChange={handleUpload}
          className="hidden"
        />

        <button
          onClick={triggerUpload}
          className="bg-indigo-600 hover:bg-indigo-500 transition px-6 py-3 rounded-lg font-semibold shadow-lg"
        >
          Upload Image or Video
        </button>

        <p className="text-gray-400 text-sm mt-2">
          Supported formats: JPG, PNG, MP4
        </p>
      </div>

      {/* Preview + Output */}
      <div className="grid grid-cols-2 gap-6 w-full max-w-5xl">
        <div className="border border-gray-700 rounded-lg p-4">
          <h2 className="mb-2 font-semibold">Input</h2>

          {preview ? (
            file.type.startsWith("video") ? (
              <video src={preview} controls className="w-full rounded" />
            ) : (
              <img src={preview} className="w-full rounded" />
            )
          ) : (
            <p className="text-gray-400">No file uploaded</p>
          )}
        </div>

        <div className="border border-gray-700 rounded-lg p-4">
          <h2 className="mb-2 font-semibold">Output</h2>

          {loading && <p className="text-blue-400">Processing...</p>}

          {!loading && output && (
            <>
              {selectedTool === "Text Detect" ? (
                <div className="bg-gray-800 p-4 rounded max-h-64 overflow-y-auto">
                  <h3 className="font-semibold mb-2">Extracted Text:</h3>
                  <pre className="text-sm whitespace-pre-wrap">{output}</pre>
                </div>
              ) : output.startsWith("data:video") ? (
                <video src={output} controls className="w-full rounded" />
              ) : (
                <img src={output} className="w-full rounded" />
              )}

              {selectedTool !== "Text Detect" && (
                <button
                  onClick={downloadFile}
                  className="mt-3 bg-green-600 px-4 py-2 rounded hover:bg-green-500"
                >
                  Download
                </button>
              )}
            </>
          )}

          {!loading && !output && (
            <p className="text-gray-400">Output will appear here</p>
          )}
        </div>
      </div>

      {/* Tool Buttons */}
      {file && (
        <div className="mt-8 flex flex-wrap gap-4 justify-center">
          {tools.map((tool) => (
            <button
              key={tool}
              onClick={() => runTool(tool)}
              className="bg-blue-600 px-4 py-2 rounded hover:bg-blue-500"
            >
              {(tool === "Compress" && selectedTool === "Compress") ||
              (tool === "Brighten" && selectedTool === "Brighten")
                ? `Run ${tool}`
                : tool}
            </button>
          ))}
        </div>
      )}

      {/* Compression Slider */}
      {selectedTool === "Compress" && (
        <div className="mt-6 w-96">
          <label className="block mb-2">Compression: {compression}%</label>

          <input
            type="range"
            min="10"
            max="100"
            value={compression}
            onChange={(e) => setCompression(e.target.value)}
            className="w-full"
          />
        </div>
      )}

      {/* Brighten Slider */}
      {selectedTool === "Brighten" && (
        <div className="mt-6 w-96">
          <label className="block mb-2">Intensity: {lowlightIntensity}</label>

          <input
            type="range"
            min="1.0"
            max="5.0"
            step="0.1"
            value={lowlightIntensity}
            onChange={(e) => setLowlightIntensity(parseFloat(e.target.value))}
            className="w-full"
          />
        </div>
      )}
    </div>
  );
}
