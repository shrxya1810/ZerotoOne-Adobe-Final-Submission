import {
  Upload,
  FileText,
  Highlighter,
  Sparkles,
  Network,
  X,
} from "lucide-react";
import React, { useState, useEffect } from "react";

const steps = [
  {
    label: "Upload Past Docs",
    icon: <Upload className="h-5 w-5" />,
  },
  {
    label: "Upload Current Doc",
    icon: <FileText className="h-5 w-5" />,
  },
  {
    label: "Select Text",
    icon: <Highlighter className="h-5 w-5" />,
  },
  {
    label: "Explore Insights & Audio",
    icon: <Sparkles className="h-5 w-5" />,
  },
  {
    label: "Knowledge Graph",
    icon: <Network className="h-5 w-5" />,
  },
];

export type JourneyStep = 0 | 1 | 2 | 3 | 4;

interface JourneyBarProps {
  currentStep: JourneyStep;
}

export const JourneyBar: React.FC<JourneyBarProps> = ({ currentStep }) => {
  const [isVisible, setIsVisible] = useState(true);

  useEffect(() => {
    const isHidden = localStorage.getItem("journeyBarHidden");
    if (isHidden === "true") {
      setIsVisible(false);
    }
  }, []);

  const handleClose = () => {
    setIsVisible(false);
    localStorage.setItem("journeyBarHidden", "true");
  };

  if (!isVisible) {
    return null;
  }

  return (
    <nav className="sticky top-0 z-50 bg-slate-900/90 backdrop-blur border-b border-slate-800 py-2 px-4 flex items-center justify-center gap-8 shadow-md relative">
      {/* Close Button */}
      <button
        onClick={handleClose}
        className="absolute right-4 top-1/2 transform -translate-y-1/2 p-1 rounded-full hover:bg-slate-700/50 transition-colors duration-200 group"
        title="Hide progress steps"
      >
        <X className="h-4 w-4 text-slate-400 group-hover:text-slate-200" />
      </button>

      {steps.map((step, idx) => (
        <div key={step.label} className="flex flex-col items-center">
          <div
            className={`flex items-center justify-center rounded-full w-10 h-10 mb-1 text-lg font-bold transition-all duration-300
              ${idx < currentStep ? "bg-red-400 text-white" : idx === currentStep ? "bg-white text-red-500 ring-2 ring-red-400" : "bg-slate-800 text-slate-400"}`}
          >
            {step.icon}
            <span className="ml-1">{idx + 1}</span>
          </div>
          <span
            className={`text-xs font-medium ${idx <= currentStep ? "text-red-400" : "text-slate-400"}`}
          >
            {step.label}
          </span>
        </div>
      ))}
    </nav>
  );
};

export default JourneyBar;
