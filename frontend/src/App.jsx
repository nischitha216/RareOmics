
import { useEffect, useState } from "react";
import { jsPDF } from "jspdf";

import {
  Dna,
  Upload,
  FileSpreadsheet,
  Activity,
  BrainCircuit,
  Microscope,
  CheckCircle2,
  AlertCircle,
  LoaderCircle,
  ArrowRight,
  RotateCcw,
  FileText,
  ShieldCheck,
  ChartNoAxesCombined,
  Lightbulb,
  History,
  Trash2,
  Clock3,
} from "lucide-react";

import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
} from "recharts";

import "./App.css";

const API_URL = "http://127.0.0.1:8000";

const subtypeColors = {
  "Kinase signaling": "#18866b",
  "Wnt-altered": "#3979c5",
  "Pseudohypoxia": "#b78338",
  "Cortical admixture": "#8965b5",
};

function App() {
  const [genomicsFile, setGenomicsFile] = useState(null);
  const [epigenomicsFile, setEpigenomicsFile] = useState(null);

  const [model, setModel] = useState("logistic_regression");
  const [results, setResults] = useState(null);
  const [predictionHistory, setPredictionHistory] = useState(() => {
    try {
      return JSON.parse(sessionStorage.getItem("rareomics_prediction_history") || "[]");
    } catch {
      return [];
    }
  });
  const [comparisonResults, setComparisonResults] = useState(null);
  const [comparing, setComparing] = useState(false);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [activePage, setActivePage] = useState("prediction");
  const [showHistoryPanel, setShowHistoryPanel] = useState(false);
  const [validation, setValidation] = useState(null);
  const [validating, setValidating] = useState(false);

  // Keep prediction history for this browser session only.
  useEffect(() => {
    try {
      sessionStorage.setItem(
        "rareomics_prediction_history",
        JSON.stringify(predictionHistory)
      );
    } catch (storageError) {
      console.warn("Could not save prediction history for this session.", storageError);
    }
  }, [predictionHistory]);

  const addToHistory = (predictionData) => {
    const entry = {
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      timestamp: new Date().toISOString(),
      model_used: predictionData.model_used,
      samples_predicted: predictionData.samples_predicted,
      prediction_target: predictionData.prediction_target,
      predictions: predictionData.predictions,
    };
    setPredictionHistory((previous) => [entry, ...previous].slice(0, 20));
  };

  // ---------------- PREDICTION ----------------

  const handlePredict = async () => {
    setError("");
    setResults(null);

    if (!genomicsFile || !epigenomicsFile) {
      setError("Please upload both genomics and epigenomics CSV files.");
      return;
    }

    const formData = new FormData();

    formData.append("genomics_file", genomicsFile);
    formData.append("epigenomics_file", epigenomicsFile);

    setLoading(true);

    try {
      const response = await fetch(
        `${API_URL}/predict/multiomics?model_name=${model}`,
        {
          method: "POST",
          body: formData,
        }
      );

      const data = await response.json();

      if (!response.ok) {
        const detail = Array.isArray(data.detail)
          ? data.detail.map((item) => item.msg).join(", ")
          : data.detail;

        throw new Error(detail || "Prediction request failed.");
      }

      setResults(data);
      addToHistory(data);
    } catch (err) {
      setError(
        err.message ||
          "Unable to connect to the backend. Check that FastAPI is running."
      );
    } finally {
      setLoading(false);
    }
  };

  // Run both saved models against the same uploaded datasets
  const handleCompareModels = async () => {
    setError("");
    setComparisonResults(null);

    if (!genomicsFile || !epigenomicsFile) {
      setError("Please upload both genomics and epigenomics CSV files first.");
      return;
    }

    setComparing(true);

    try {
      const runModel = async (modelName) => {
        const formData = new FormData();
        formData.append("genomics_file", genomicsFile);
        formData.append("epigenomics_file", epigenomicsFile);

        const response = await fetch(
          `${API_URL}/predict/multiomics?model_name=${modelName}`,
          { method: "POST", body: formData }
        );
        const data = await response.json();

        if (!response.ok) {
          const detail = Array.isArray(data.detail)
            ? data.detail.map((item) => item.msg).join(", ")
            : data.detail;
          throw new Error(detail || `${modelName} prediction failed.`);
        }
        return data;
      };

      const [logistic, forest] = await Promise.all([
        runModel("logistic_regression"),
        runModel("random_forest"),
      ]);

      setComparisonResults({ logistic, forest });
      addToHistory(logistic);
      addToHistory(forest);
    } catch (err) {
      setError(err.message || "Model comparison failed.");
    } finally {
      setComparing(false);
    }
  };

  // ---------------- RESET ----------------

  const resetResults = () => {
    setResults(null);
    setComparisonResults(null);
    setValidation(null);
    setError("");
  };

  // ---------------- DOWNLOAD FORMAL PDF SUMMARY REPORT ----------------

  const downloadSummaryReport = () => {
    if (!results?.predictions?.length) return;

    const doc = new jsPDF({ unit: "mm", format: "a4" });
    const pageWidth = doc.internal.pageSize.getWidth();
    const pageHeight = doc.internal.pageSize.getHeight();
    const margin = 18;
    const contentWidth = pageWidth - margin * 2;
    const subtypeDescriptions = {
      "Kinase signaling": "A PCPG molecular subtype characterised by a gene-expression pattern associated with kinase-signalling pathways.",
      "Wnt-altered": "A PCPG molecular subtype associated with alterations in Wnt-related signalling patterns.",
      "Pseudohypoxia": "A PCPG molecular subtype associated with hypoxia-related signalling, including pathways that can remain active under oxygen-available conditions.",
      "Cortical admixture": "A PCPG molecular subtype whose expression profile shows a cortical-like or adrenal-cortex-associated component."
    };
    let y = 22;

    const drawSectionHeading = (label) => {
      ensureSpace(14);
      doc.setFillColor(232, 243, 237);
      doc.roundedRect(margin, y - 5, contentWidth, 9, 1.5, 1.5, "F");
      doc.setTextColor(30, 91, 67);
      doc.setFont("helvetica", "bold");
      doc.setFontSize(9);
      doc.text(label.toUpperCase(), margin + 3, y + 1);
      y += 13;
    };

    const ensureSpace = (height) => {
      if (y + height > pageHeight - 19) {
        doc.addPage();
        y = 22;
      }
    };

    const drawParagraph = (text, options = {}) => {
      const { size = 9, style = "normal", color = [65, 76, 70], gap = 3, indent = 0 } = options;
      doc.setFont("helvetica", style);
      doc.setFontSize(size);
      doc.setTextColor(...color);
      const lines = doc.splitTextToSize(String(text), contentWidth - indent);
      const lineHeight = size * 0.48;
      ensureSpace(lines.length * lineHeight + gap);
      doc.text(lines, margin + indent, y);
      y += lines.length * lineHeight + gap;
    };

    const drawKeyValue = (label, value) => {
      ensureSpace(7);
      doc.setFont("helvetica", "bold");
      doc.setFontSize(9);
      doc.setTextColor(75, 88, 80);
      doc.text(`${label}:`, margin, y);
      doc.setFont("helvetica", "normal");
      doc.setTextColor(45, 55, 49);
      doc.text(String(value), margin + 43, y);
      y += 6.5;
    };

    // Formal report header
    doc.setFillColor(21, 59, 50);
    doc.rect(0, 0, pageWidth, 35, "F");
    doc.setTextColor(255, 255, 255);
    doc.setFont("helvetica", "bold");
    doc.setFontSize(21);
    doc.text("RareOmics", margin, 16);
    doc.setFont("helvetica", "normal");
    doc.setFontSize(9);
    doc.text("MULTI-OMICS MOLECULAR SUBTYPE ANALYSIS", margin, 24);
    doc.setTextColor(45, 55, 49);
    y = 46;

    doc.setFont("helvetica", "bold");
    doc.setFontSize(15);
    doc.text("Prediction Summary Report", margin, y);
    y += 9;
    drawParagraph("Pheochromocytoma and Paraganglioma (PCPG)", { size: 10, color: [90, 103, 95], gap: 7 });

    drawSectionHeading("Report information");
    drawKeyValue("Generated", new Date().toLocaleString());
    drawKeyValue("Model", formatModel(results.model_used));
    drawKeyValue("Samples predicted", results.samples_predicted);
    drawKeyValue("Prediction target", results.prediction_target || "PCPG mRNA molecular subtype cluster");
    y += 3;

    drawSectionHeading("Prediction results");
    results.predictions.forEach((patient, index) => {
      const subtype = patient.predicted_subtype;
      const confidence = Number(patient.probabilities?.[subtype] ?? 0) * 100;
      ensureSpace(39);
      doc.setDrawColor(218, 229, 221);
      doc.setFillColor(250, 252, 250);
      doc.roundedRect(margin, y - 4, contentWidth, 31, 2, 2, "FD");
      doc.setFont("helvetica", "bold");
      doc.setFontSize(9);
      doc.setTextColor(42, 72, 56);
      doc.text(`SAMPLE ${index + 1}`, margin + 4, y + 2);
      doc.setFont("helvetica", "normal");
      doc.setTextColor(55, 65, 59);
      doc.text(`Sample ID: ${String(patient.patient_id)}`, margin + 4, y + 8);
      doc.setFont("helvetica", "bold");
      doc.setTextColor(27, 111, 79);
      doc.text(`Predicted subtype: ${String(subtype)}`, margin + 4, y + 15);
      doc.setFont("helvetica", "normal");
      doc.setTextColor(65, 76, 70);
      doc.text(`Highest class probability: ${confidence.toFixed(2)}%`, margin + 4, y + 22);
      y += 34;

      drawParagraph(subtypeDescriptions[subtype] || "A model-assigned PCPG molecular subtype category.", { size: 8.5, color: [75, 85, 78], indent: 2, gap: 3 });

      ensureSpace(12);
      doc.setFont("helvetica", "bold");
      doc.setFontSize(8);
      doc.setTextColor(75, 88, 80);
      doc.text("CLASS PROBABILITIES", margin + 2, y);
      y += 5;
      const entries = Object.entries(patient.probabilities || {});
      const col1 = margin + 3;
      const col2 = margin + contentWidth - 3;
      entries.forEach(([className, value]) => {
        ensureSpace(6);
        doc.setFont("helvetica", "normal");
        doc.setFontSize(8.5);
        doc.setTextColor(55, 65, 59);
        doc.text(String(className), col1, y);
        doc.text(`${(Number(value) * 100).toFixed(2)}%`, col2, y, { align: "right" });
        y += 5.5;
      });
      y += 4;
    });

    // Consistent footer and page numbering on every page
    const pageCount = doc.getNumberOfPages();
    for (let page = 1; page <= pageCount; page += 1) {
      doc.setPage(page);
      doc.setDrawColor(218, 229, 221);
      doc.line(margin, pageHeight - 14, pageWidth - margin, pageHeight - 14);
      doc.setFont("helvetica", "normal");
      doc.setFontSize(7.5);
      doc.setTextColor(115, 127, 119);
      doc.text("RareOmics | Research summary — not for clinical use", margin, pageHeight - 8);
      doc.text(`Page ${page} of ${pageCount}`, pageWidth - margin, pageHeight - 8, { align: "right" });
    }

    doc.save(`RareOmics_Prediction_Summary_${new Date().toISOString().slice(0, 10)}.pdf`);
  };

  // ---------------- HELPERS ----------------

  const formatModel = (value) =>
    value === "logistic_regression"
      ? "Logistic Regression"
      : "Random Forest";

  const getProbabilityData = (probabilities) =>
    Object.entries(probabilities || {}).map(([name, value]) => ({
      name,
      probability: Number((value * 100).toFixed(2)),
      fill: subtypeColors[name] || "#18866b",
    }));

  // ---------------- DATA VALIDATION ----------------

  const parseCSV = (text) => {
    const rows = [];
    let row = [];
    let cell = "";
    let quoted = false;
    for (let i = 0; i < text.length; i += 1) {
      const char = text[i];
      if (char === '"' && quoted && text[i + 1] === '"') {
        cell += '"';
        i += 1;
      } else if (char === '"') {
        quoted = !quoted;
      } else if (char === "," && !quoted) {
        row.push(cell);
        cell = "";
      } else if ((char === "\n" || char === "\r") && !quoted) {
        if (char === "\r" && text[i + 1] === "\n") i += 1;
        row.push(cell);
        if (row.some((value) => value.trim() !== "")) rows.push(row);
        row = [];
        cell = "";
      } else {
        cell += char;
      }
    }
    row.push(cell);
    if (row.some((value) => value.trim() !== "")) rows.push(row);
    return rows;
  };

  const validateDatasets = async () => {
    setError(""); setValidation(null);
    if (!genomicsFile || !epigenomicsFile) {
      setError("Upload both CSV files before checking missing or duplicate values."); return;
    }
    setValidating(true);
    try {
      const [gText, eText] = await Promise.all([genomicsFile.text(), epigenomicsFile.text()]);
      const inspect = (text, label) => {
        const rows = parseCSV(text);
        if (rows.length < 2) throw new Error(`${label}: file has no sample rows.`);
        const headers = rows[0].map((v) => v.trim().replace(/^\uFEFF/, ""));
        const data = rows.slice(1);
        const ids = data.map((r) => (r[0] || "").trim());
        let missing = 0;
        data.forEach((r) => headers.forEach((_, i) => { if (!String(r[i] ?? "").trim()) missing += 1; }));
        const duplicateIds = ids.filter((v, i) => v && ids.indexOf(v) !== i).length;
        const featureNames = headers.slice(1);
        const duplicateHeaders = featureNames.filter((v, i) => v && featureNames.indexOf(v) !== i).length;
        return { label, samples: data.length, missing, duplicateIds, duplicateHeaders };
      };
      const g = inspect(gText, "Genomics"); const e = inspect(eText, "Epigenomics");
      const missingTotal = g.missing + e.missing;
      const duplicateIdsTotal = g.duplicateIds + e.duplicateIds;
      const duplicateHeadersTotal = g.duplicateHeaders + e.duplicateHeaders;
      const checks = [
        { name: "Missing values", pass: missingTotal === 0, detail: `${missingTotal.toLocaleString()} blank cells across both files` },
        { name: "Duplicate sample IDs", pass: duplicateIdsTotal === 0, detail: `${duplicateIdsTotal} duplicates across both files` },
        { name: "Duplicate feature names", pass: duplicateHeadersTotal === 0, detail: `${duplicateHeadersTotal} duplicates across both files` }
      ];
      setValidation({ checks, genomics: g, epigenomics: e, missingTotal, duplicateIdsTotal, duplicateHeadersTotal, passed: checks.filter(c => c.pass).length, total: checks.length, ready: checks.every(c => c.pass) });
    } catch (err) {
      setError(err.message || "Could not check the uploaded CSV files.");
    } finally { setValidating(false); }
  };

  // ---------------- FILE UPLOAD ----------------

  const renderUpload = (
    title,
    description,
    file,
    setFile,
    id,
    featureCount
  ) => (
    <div className="upload-card">
      <div className="upload-heading">
        <div className="upload-icon">
          <FileSpreadsheet size={22} />
        </div>

        <div>
          <h3>{title}</h3>
          <p>{description}</p>
        </div>
      </div>

      <label className="drop-zone" htmlFor={id}>
        <Upload size={25} />

        <strong>
          {file ? file.name : "Choose CSV file"}
        </strong>

        <span>
          {file
            ? `${(file.size / 1024).toFixed(1)} KB`
            : "Click to browse your files"}
        </span>

        <input
          id={id}
          type="file"
          accept=".csv,text/csv"
          onChange={(e) => {
            const selected = e.target.files?.[0];

            if (
              selected &&
              !selected.name.toLowerCase().endsWith(".csv")
            ) {
              setError("Please select a CSV file.");
              e.target.value = "";
              return;
            }

            setError("");
            setFile(selected || null);
            setResults(null);
            setComparisonResults(null);
            setValidation(null);
          }}
        />
      </label>

      <div className="upload-footer">
        <span>{featureCount} features expected</span>

        {file && (
          <span className="file-ready">
            <CheckCircle2 size={14} /> Ready
          </span>
        )}
      </div>
    </div>
  );

  // ---------------- SIDEBAR ----------------

  const renderSidebar = () => (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-icon">
          <Dna size={27} />
        </div>

        <div>
          <h2>RareOmics</h2>
          <span>MULTI-OMICS PLATFORM</span>
        </div>
      </div>

      <div className="sidebar-label">WORKSPACE</div>

      <button
        className={`nav-item ${
          activePage === "prediction" ? "active" : ""
        }`}
        onClick={() => setActivePage("prediction")}
      >
        <Activity size={19} />
        <span>Subtype Prediction</span>
      </button>

      <button
        className={`nav-item ${
          activePage === "analysis" ? "active" : ""
        }`}
        onClick={() => setActivePage("analysis")}
      >
        <BrainCircuit size={19} />
        <span>Multi-Omics Analysis</span>
      </button>

      <button className={`nav-item ${activePage === "explainability" ? "active" : ""}`} onClick={() => setActivePage("explainability")}>
        <Lightbulb size={19} /><span>Explainability</span>
      </button>
      <button className={`nav-item ${activePage === "evaluation" ? "active" : ""}`} onClick={() => setActivePage("evaluation")}>
        <ChartNoAxesCombined size={19} /><span>Model Evaluation</span>
      </button>

      <div className="sidebar-bottom">
        <div className="sidebar-status">
          <span className="status-dot" />
          <span>Prediction workspace</span>
        </div>

        <p>Research prototype · v1.0</p>
      </div>
    </aside>
  );

  // ---------------- HEADER ----------------

  const renderHeader = () => (
    <header className="topbar">
      <div>
        <span className="breadcrumb">
          WORKSPACE /{" "}
          {{ prediction: "PREDICTION", analysis: "MULTI-OMICS ANALYSIS", explainability: "MODEL EXPLAINABILITY", evaluation: "MODEL EVALUATION" }[activePage]}
        </span>

        <h1>
          {{ prediction: "Molecular Subtype Prediction", analysis: "Multi-Omics Analysis", explainability: "Model Explainability", evaluation: "Model Evaluation" }[activePage]}
        </h1>
      </div>

      <div className="topbar-actions">
        <div className="history-control">
          <button
            className="history-trigger"
            onClick={() => setShowHistoryPanel((open) => !open)}
            aria-expanded={showHistoryPanel}
            aria-label="Open prediction history"
          >
            <History size={16} />
            <span>History</span>
            <span className="history-count">{predictionHistory.length}</span>
          </button>
          {showHistoryPanel && (
            <div className="history-popover">
              <div className="history-popover-heading">
                <div><strong>Recent predictions</strong><small>Saved in this browser session</small></div>
                {predictionHistory.length > 0 && (
                  <button className="history-clear" onClick={() => setPredictionHistory([])}>
                    <Trash2 size={14} /> Clear
                  </button>
                )}
              </div>
              {predictionHistory.length === 0 ? (
                <div className="history-popover-empty">
                  <Clock3 size={22} />
                  <strong>No predictions yet</strong>
                  <span>Your successful prediction runs will appear here.</span>
                </div>
              ) : (
                <div className="history-popover-list">
                  {predictionHistory.map((entry) => (
                    <div className="history-popover-item" key={entry.id}>
                      <div className="history-item-info">
                        <strong>{formatModel(entry.model_used)}</strong>
                        <span>{entry.samples_predicted} samples · {new Date(entry.timestamp).toLocaleString()}</span>
                      </div>
                      <button className="history-view-button" onClick={() => {
                        setResults({
                          status: "success",
                          model_used: entry.model_used,
                          samples_predicted: entry.samples_predicted,
                          prediction_target: entry.prediction_target,
                          predictions: entry.predictions,
                        });
                        setComparisonResults(null);
                        setActivePage("prediction");
                        setShowHistoryPanel(false);
                      }}>View</button>
                    </div>
                  ))}
                </div>
              )}
              <p className="history-popover-note">Prediction outputs only; uploaded CSV files are not stored.</p>
            </div>
          )}
        </div>
        <div className="topbar-tag"><Microscope size={16} />Research Platform</div>
      </div>
    </header>
  );

  // ---------------- PREDICTION PAGE ----------------

  const renderPredictionPage = () => (
    <>
      <section className="intro">
        <div>
          <h2>Multi-Omics Analysis</h2>

          <p>
            Integrate genomic and epigenomic features to predict
            molecular subtypes of pheochromocytoma and
            paraganglioma (PCPG).
          </p>
        </div>

        <div className="intro-icon">
          <Dna size={42} />
        </div>
      </section>

      <section className="stats-grid">
        <div className="stat-card">
          <span>GENOMIC FEATURES</span>
          <strong>43</strong>
          <small>Processed genomic inputs</small>
        </div>

        <div className="stat-card">
          <span>EPIGENOMIC FEATURES</span>
          <strong>20,113</strong>
          <small>Methylation inputs</small>
        </div>

        <div className="stat-card">
          <span>INTEGRATED FEATURES</span>
          <strong>20,156</strong>
          <small>Combined model inputs</small>
        </div>

        <div className="stat-card">
          <span>MOLECULAR SUBTYPES</span>
          <strong>04</strong>
          <small>Prediction classes</small>
        </div>
      </section>

      {/* UPLOAD SECTION */}

      <section className="section-card">
        <div className="section-heading">
          <div>
            <span className="step-label">STEP 01</span>
            <h2>Upload datasets</h2>
            <p>Upload your processed CSV feature matrices.</p>
          </div>

          <span className="required-label">
            Both files required
          </span>
        </div>

        <div className="upload-grid">
          {renderUpload(
            "Genomics",
            "Genomic feature matrix",
            genomicsFile,
            setGenomicsFile,
            "genomics-upload",
            "43"
          )}

          {renderUpload(
            "Epigenomics",
            "DNA methylation feature matrix",
            epigenomicsFile,
            setEpigenomicsFile,
            "epigenomics-upload",
            "20,113"
          )}
        </div>

        <div className="upload-note">
          <AlertCircle size={17} />

          <span>
            Upload processed CSV files with patient IDs in the
            first column. Raw FASTQ, VCF, BAM and IDAT files
            are not accepted.
          </span>
        </div>
        <button className="validate-button" onClick={validateDatasets} disabled={validating || !genomicsFile || !epigenomicsFile}>
          {validating ? <><LoaderCircle className="spin" size={18} /> Validating files...</> : <><ShieldCheck size={18} />Data Validation</>}
        </button>
        {validation && <div className="validation-panel">
          <div className="validation-summary"><strong>{validation.passed}/{validation.total} checks passed</strong><span>{validation.ready ? "Missing-value and duplicate checks passed." : "Review the detected missing or duplicate values before prediction."}</span></div>
          {validation.checks.map((check) => <div className="validation-row" key={check.name}><span className={check.pass ? "validation-pass" : "validation-fail"}>{check.pass ? "✓" : "!"}</span><div><strong>{check.name}</strong><small>{check.detail}</small></div></div>)}
        </div>}
      </section>

      {/* MODEL SELECTION */}

      <section className="section-card">
        <div className="section-heading">
          <div>
            <span className="step-label">STEP 02</span>
            <h2>Select prediction model</h2>
            <p>Choose the trained machine learning pipeline.</p>
          </div>
        </div>

        <div className="model-grid">
          <button
            className={`model-option ${
              model === "logistic_regression" ? "selected" : ""
            }`}
            onClick={() => setModel("logistic_regression")}
          >
            <BrainCircuit size={23} />

            <div>
              <strong>Logistic Regression</strong>
              <span>Linear classification model</span>
            </div>

            {model === "logistic_regression" && (
              <CheckCircle2 className="model-check" size={19} />
            )}
          </button>

          <button
            className={`model-option ${
              model === "random_forest" ? "selected" : ""
            }`}
            onClick={() => setModel("random_forest")}
          >
            <Activity size={23} />

            <div>
              <strong>Random Forest</strong>
              <span>Ensemble classification model</span>
            </div>

            {model === "random_forest" && (
              <CheckCircle2 className="model-check" size={19} />
            )}
          </button>
        </div>

        <button
          className="predict-button"
          onClick={handlePredict}
          disabled={loading}
        >
          {loading ? (
            <>
              <LoaderCircle className="spin" size={19} />
              Generating predictions...
            </>
          ) : (
            <>
              Run Subtype Prediction
              <ArrowRight size={19} />
            </>
          )}
        </button>

        <button
          className="compare-button"
          onClick={handleCompareModels}
          disabled={comparing || loading}
        >
          {comparing ? (
            <><LoaderCircle className="spin" size={18} /> Comparing both models...</>
          ) : (
            <>Compare Both Models <ArrowRight size={18} /></>
          )}
        </button>

        {error && (
          <div className="error-message">
            <AlertCircle size={18} />
            {error}
          </div>
        )}
      </section>

      {/* RESULTS */}

      {results && (
        <section className="section-card results-section">
          <div className="section-heading">
            <div>
              <span className="step-label">STEP 03</span>
              <h2>Prediction results</h2>
              <p>Model inference completed successfully.</p>
            </div>

            <div className="result-actions">
              <button
                className="download-button"
                onClick={downloadSummaryReport}
              >
                <FileText size={16} />
                Download Summary Report
              </button>

              <button
                className="reset-button"
                onClick={resetResults}
              >
                <RotateCcw size={16} />
                Clear results
              </button>
            </div>
          </div>

          <div className="result-summary">
            <div>
              <span>MODEL USED</span>
              <strong>{formatModel(results.model_used)}</strong>
            </div>

            <div>
              <span>SAMPLES PREDICTED</span>
              <strong>{results.samples_predicted}</strong>
            </div>

            <div>
              <span>TARGET</span>
              <strong>{results.prediction_target}</strong>
            </div>
          </div>

          {results.predictions?.map((patient, index) => {
            const probabilityData = getProbabilityData(
              patient.probabilities
            );

            const winningProbability = Math.max(
              ...probabilityData.map(
                (item) => item.probability
              )
            );

            return (
              <div
                className="patient-result"
                key={patient.patient_id}
              >
                <div className="patient-heading">
                  <div>
                    <span className="patient-number">
                      PATIENT {index + 1}
                    </span>

                    <h3>{patient.patient_id}</h3>
                  </div>

                  <span className="result-badge">
                    <CheckCircle2 size={15} />
                    Predicted
                  </span>
                </div>

                <div className="prediction-highlight">
                  <span>PREDICTED MOLECULAR SUBTYPE</span>

                  <h2>{patient.predicted_subtype}</h2>

                  <p>
                    Highest model probability: {winningProbability}%
                  </p>
                </div>

                <h4>Subtype probability distribution</h4>

                <div className="chart-container">
                  <ResponsiveContainer width="100%" height={230}>
                    <BarChart
                      data={probabilityData}
                      layout="vertical"
                      margin={{
                        top: 5,
                        right: 25,
                        left: 15,
                        bottom: 5,
                      }}
                    >
                      <CartesianGrid
                        strokeDasharray="3 3"
                        horizontal={false}
                        stroke="#e9eeeb"
                      />

                      <XAxis
                        type="number"
                        domain={[0, 100]}
                        unit="%"
                        tick={{
                          fill: "#74827c",
                          fontSize: 11,
                        }}
                      />

                      <YAxis
                        type="category"
                        dataKey="name"
                        width={125}
                        tick={{
                          fill: "#46564f",
                          fontSize: 11,
                        }}
                      />

                      <Tooltip
                        formatter={(value) => [
                          `${value}%`,
                          "Probability",
                        ]}
                      />

                      <Bar
                        dataKey="probability"
                        radius={[0, 5, 5, 0]}
                        fill="#18866b"
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            );
          })}
        </section>
      )}

      {comparisonResults && (
        <section className="section-card results-section">
          <div className="section-heading">
            <div>
              <span className="step-label">MODEL COMPARISON</span>
              <h2>Logistic Regression vs Random Forest</h2>
            </div>
          </div>

          <div className="comparison-grid">
            {[
              { label: "Logistic Regression", data: comparisonResults.logistic },
              { label: "Random Forest", data: comparisonResults.forest },
            ].map((item) => (
              <div className="comparison-card" key={item.label}>
                <h3>{item.label}</h3>
                <p>{item.data.samples_predicted} samples predicted</p>
                {item.data.predictions.map((patient) => (
                  <div className="comparison-patient" key={patient.patient_id}>
                    <span>{patient.patient_id}</span>
                    <strong>{patient.predicted_subtype}</strong>
                  </div>
                ))}
              </div>
            ))}
          </div>

        </section>
      )}

      <footer className="footer">
        <div>
          <Dna size={17} />
          <strong>RareOmics</strong>
        </div>

        <span>
          Research prototype · Not intended for clinical diagnosis
        </span>
      </footer>
    </>
  );

  // ---------------- MULTI-OMICS ANALYSIS PAGE ----------------

  const renderAnalysisPage = () => (
    <>
      <section className="intro">
        <div>
          <h2>Understanding the Multi-Omics Workflow</h2>

          <p>
            An integrated machine learning approach combining
            genomic and epigenomic information for PCPG
            molecular subtype classification.
          </p>
        </div>

        <div className="intro-icon">
          <Dna size={42} />
        </div>
      </section>

      <section className="stats-grid">
        <div className="stat-card">
          <span>GENOMIC FEATURES</span>
          <strong>43</strong>
          <small>Genomic input variables</small>
        </div>

        <div className="stat-card">
          <span>EPIGENOMIC FEATURES</span>
          <strong>20,113</strong>
          <small>Methylation input variables</small>
        </div>

        <div className="stat-card">
          <span>TRAINING SAMPLES</span>
          <strong>173</strong>
          <small>Integrated model training samples</small>
        </div>

        <div className="stat-card">
          <span>SUBTYPE CLASSES</span>
          <strong>04</strong>
          <small>Classification categories</small>
        </div>
      </section>

      <section className="section-card">
        <div className="section-heading">
          <div>
            <span className="step-label">PROJECT OVERVIEW</span>
            <h2>About RareOmics</h2>
            <p>Project objective and computational approach.</p>
          </div>
        </div>

        <div className="analysis-intro">
          <h3>Research objective</h3>

          <p>
            RareOmics investigates whether integrated
            genomic and epigenomic features can predict the
            mRNA molecular subtype clusters of
            pheochromocytoma and paraganglioma (PCPG).
          </p>

          <p>
            Transcriptomic subtype labels are used as the
            prediction target. Transcriptomic features are
            not included as model input features in this
            integrated prediction workflow.
          </p>
        </div>

        <h3 className="analysis-title">Multi-omics workflow</h3>

        <div className="workflow-grid">
          <div className="workflow-card">
            <span>01</span>
            <Dna size={25} />
            <h3>Genomics</h3>
            <p>
              43 genomic features form the genomic input
              dataset.
            </p>
          </div>

          <div className="workflow-card">
            <span>02</span>
            <Activity size={25} />
            <h3>Epigenomics</h3>
            <p>
              20,113 DNA methylation features form the
              epigenomic input dataset.
            </p>
          </div>

          <div className="workflow-card">
            <span>03</span>
            <BrainCircuit size={25} />
            <h3>Feature Integration</h3>
            <p>
              Both input matrices are aligned by sample ID
              and combined into 20,156 features.
            </p>
          </div>

          <div className="workflow-card">
            <span>04</span>
            <Microscope size={25} />
            <h3>Subtype Prediction</h3>
            <p>
              Integrated features are used to predict
              mRNA molecular subtype clusters.
            </p>
          </div>
        </div>
      </section>

      <section className="section-card">
        <div className="section-heading">
          <div>
            <span className="step-label">MODEL DEVELOPMENT</span>
            <h2>Machine learning models</h2>
            <p>Two trained integrated classification pipelines.</p>
          </div>
        </div>

        <div className="model-grid">
          <div className="workflow-card">
            <BrainCircuit size={25} />

            <h3>Logistic Regression</h3>

            <p>
              Multiclass linear classifier using the integrated feature pipeline.
            </p>
          </div>

          <div className="workflow-card">
            <Activity size={25} />

            <h3>Random Forest</h3>

            <p>
              Tree-based ensemble classifier using the integrated feature pipeline.
            </p>
          </div>
        </div>

        <div className="analysis-note">
          Quantitative cross-validation metrics are available on the separate Model Evaluation page.
        </div>
      </section>

      <section className="section-card">
        <div className="section-heading">
          <div>
            <span className="step-label">CLASSIFICATION TARGET</span>
            <h2>Molecular subtype classes</h2>
            <p>Four mRNA subtype categories predicted by the models.</p>
          </div>
        </div>

        <div className="subtype-grid">
          <div>Kinase signaling</div>
          <div>Wnt-altered</div>
          <div>Pseudohypoxia</div>
          <div>Cortical admixture</div>
        </div>

        <div className="analysis-note">
          This application is a research prototype.
          Predictions are computational outputs and are not
          intended for clinical diagnosis or treatment decisions.
        </div>

        <button
          className="predict-button"
          onClick={() => setActivePage("prediction")}
        >
          Go to Subtype Prediction
          <ArrowRight size={18} />
        </button>
      </section>

      <footer className="footer">
        <div>
          <Dna size={17} />
          <strong>RareOmics</strong>
        </div>

        <span>Multi-Omics Research Platform · v1.0</span>
      </footer>
    </>
  );

  // ---------------- EXPLAINABILITY PAGE ----------------
  const shapFeatures = [
    { feature: "EPI_DUSP6", value: 1.379, frequency: "100%" },
    { feature: "EPI_MYBPC3", value: 1.229, frequency: "100%" },
    { feature: "EPI_EIF5B", value: 1.116, frequency: "60%" },
    { feature: "EPI_GAPT", value: 1.107, frequency: "20%" },
    { feature: "EPI_FAM150B", value: 0.967, frequency: "20%" },
    { feature: "EPI_USP2", value: 0.962, frequency: "20%" },
    { feature: "EPI_CREB5", value: 0.910, frequency: "100%" },
    { feature: "EPI_SEMA3G", value: 0.850, frequency: "20%" },
    { feature: "EPI_MIR548F2", value: 0.836, frequency: "20%" },
    { feature: "EPI_GABRG2", value: 0.812, frequency: "20%" },
  ];
  const renderExplainabilityPage = () => (
    <>
      <section className="intro"><div><h2>Which features influenced the model globally?</h2><p>Global SHAP feature importance from the integrated Logistic Regression out-of-fold analysis.</p></div><div className="intro-icon"><Lightbulb size={42}/></div></section>
      <section className="stats-grid"><div className="stat-card"><span>EXPLAINED MODEL</span><strong>Logistic</strong><small>Integrated classifier</small></div><div className="stat-card"><span>METHOD</span><strong>SHAP</strong><small>LinearExplainer</small></div><div className="stat-card"><span>VALIDATION</span><strong>5-fold</strong><small>Stratified out-of-fold</small></div><div className="stat-card"><span>FEATURES SHOWN</span><strong>10</strong><small>Top global features</small></div></section>
      <section className="section-card"><div className="section-heading"><div><span className="step-label">GLOBAL EXPLANATION</span><h2>Top 10 features by mean absolute SHAP value</h2><p>Higher values indicate a larger average contribution magnitude in the analysed folds.</p></div></div><div className="chart-container explain-chart"><ResponsiveContainer width="100%" height={360}><BarChart data={shapFeatures} layout="vertical" margin={{top:5,right:25,left:35,bottom:5}}><CartesianGrid strokeDasharray="3 3" horizontal={false}/><XAxis type="number"/><YAxis type="category" dataKey="feature" width={125}/><Tooltip formatter={(v)=>[Number(v).toFixed(3),"Mean |SHAP|"]}/><Bar dataKey="value" fill="#238466" radius={[0,5,5,0]}/></BarChart></ResponsiveContainer></div><div className="table-wrap"><table className="data-table"><thead><tr><th>Feature</th><th>Omics type</th><th>Mean |SHAP|</th><th>Fold selection frequency</th></tr></thead><tbody>{shapFeatures.map((f)=><tr key={f.feature}><td>{f.feature}</td><td>Epigenomics</td><td>{f.value.toFixed(3)}</td><td>{f.frequency}</td></tr>)}</tbody></table></div></section>
      <section className="section-card"><h2>How to interpret this</h2><p>Mean absolute SHAP measures the average magnitude of a feature's contribution to model output, without retaining whether the contribution is positive or negative. Selection frequency indicates the proportion of the five folds in which a feature was selected by the feature-selection step.</p><div className="analysis-note"></div></section>
    </>
  );

  // ---------------- MODEL EVALUATION PAGE ----------------
  const renderEvaluationPage = () => {
    const metrics = [
      { name: "Accuracy", lr: 86.10, rf: 84.94 },
      { name: "Macro F1", lr: 79.88, rf: 75.99 },
      { name: "Balanced accuracy", lr: 79.41, rf: 75.79 },
    ];
    const chartData = metrics.map((m) => ({ metric: m.name, "Logistic Regression": m.lr, "Random Forest": m.rf }));
    return <>
      <section className="intro"><div><h2>Cross-validation performance</h2><p>Evaluation metrics recorded for the integrated genomic + epigenomic classifiers.</p></div><div className="intro-icon"><ChartNoAxesCombined size={42}/></div></section>
      <section className="section-card"><div className="section-heading"><div><span className="step-label">INTEGRATED MODELS</span><h2>Performance comparison</h2><p>Five-fold cross-validation results from the saved model metadata.</p></div></div><div className="chart-container eval-chart"><ResponsiveContainer width="100%" height={330}><BarChart data={chartData} margin={{top:15,right:20,left:10,bottom:5}}><CartesianGrid strokeDasharray="3 3"/><XAxis dataKey="metric"/><YAxis domain={[0,100]} unit="%"/><Tooltip formatter={(v)=>[`${Number(v).toFixed(2)}%`, "Score"]}/><Legend/><Bar dataKey="Logistic Regression" fill="#238466" radius={[5,5,0,0]}/><Bar dataKey="Random Forest" fill="#5b83bd" radius={[5,5,0,0]}/></BarChart></ResponsiveContainer></div><div className="table-wrap"><table className="data-table"><thead><tr><th>Metric</th><th>Logistic Regression</th><th>Random Forest</th></tr></thead><tbody>{metrics.map((m)=><tr key={m.name}><td>{m.name}</td><td>{m.lr.toFixed(2)}%</td><td>{m.rf.toFixed(2)}%</td></tr>)}</tbody></table></div></section>
      <section className="section-card"><h2>Interpretation and limitations</h2><p>Logistic Regression has higher recorded cross-validation accuracy, macro F1 and balanced accuracy than Random Forest in this experiment.</p><div className="analysis-note">These are cross-validation metrics, not independent external validation. The training metadata states that both saved pipelines were fitted on all labelled samples after evaluation. Do not interpret these values as performance on unseen external cohorts.</div></section>
    </>;
  };

  // ---------------- DATA VALIDATION PAGE ----------------

  // ---------------- MAIN APP ----------------

  return (
    <div className="app-shell">
      {renderSidebar()}

      <main className="main-content">
        {renderHeader()}

        {{ prediction: renderPredictionPage, analysis: renderAnalysisPage, explainability: renderExplainabilityPage, evaluation: renderEvaluationPage }[activePage]()}
      </main>
    </div>
  );
}

export default App;
