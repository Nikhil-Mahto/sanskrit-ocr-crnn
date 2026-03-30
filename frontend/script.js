const API_URL = "http://127.0.0.1:5000/ocr";
const MAX_FILE_SIZE = 10 * 1024 * 1024;
const ACCEPTED_TYPES = new Set([
  "image/png",
  "image/jpeg",
  "image/jpg",
  "image/webp",
  "image/gif",
  "image/bmp",
]);

const imageInput = document.getElementById("imageInput");
const dropZone = document.getElementById("dropZone");
const previewFrame = document.getElementById("previewFrame");
const previewImage = document.getElementById("previewImage");
const fileMeta = document.getElementById("fileMeta");
const feedback = document.getElementById("feedback");
const runButton = document.getElementById("runButton");
const runButtonLabel = document.getElementById("runButtonLabel");
const spinner = document.getElementById("spinner");
const clearButton = document.getElementById("clearButton");
const copyButton = document.getElementById("copyButton");
const outputText = document.getElementById("outputText");
const statusBadge = document.getElementById("statusBadge");
const themeToggle = document.getElementById("themeToggle");

let selectedFile = null;

function setFeedback(message = "", type = "") {
  feedback.textContent = message;
  feedback.className = "feedback";
  if (type) {
    feedback.classList.add(`is-${type}`);
  }
}

function setStatus(message, type = "") {
  statusBadge.textContent = message;
  statusBadge.className = "status-badge";
  if (type) {
    statusBadge.classList.add(`is-${type}`);
  }
}

function setLoading(isLoading) {
  // Keep the main action button and spinner in sync with the network request state.
  runButton.disabled = isLoading || !selectedFile;
  runButton.classList.toggle("is-loading", isLoading);
  spinner.hidden = !isLoading;
  runButtonLabel.textContent = isLoading ? "Running OCR..." : "Run OCR";
}

function updateActions() {
  runButton.disabled = !selectedFile;
  copyButton.disabled = !outputText.value.trim();
}

function resetPreview() {
  previewImage.removeAttribute("src");
  previewFrame.classList.add("is-empty");
}

function resetForm() {
  selectedFile = null;
  imageInput.value = "";
  outputText.value = "";
  fileMeta.textContent = "No file selected yet.";
  setFeedback("");
  setStatus("Waiting for an image");
  resetPreview();
  updateActions();
}

function formatFileSize(bytes) {
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function validateFile(file) {
  // Validate early so the user gets instant feedback before any API request is made.
  if (!file) {
    return "No file selected.";
  }
  if (!ACCEPTED_TYPES.has(file.type)) {
    return "Please upload a valid image file (PNG, JPG, WEBP, GIF, or BMP).";
  }
  if (file.size > MAX_FILE_SIZE) {
    return "File is too large. Please choose an image under 10 MB.";
  }
  return "";
}

function previewSelectedFile(file) {
  // FileReader gives a local preview without uploading the image first.
  const reader = new FileReader();
  reader.onload = (event) => {
    previewImage.src = event.target?.result || "";
    previewFrame.classList.remove("is-empty");
  };
  reader.readAsDataURL(file);
}

function handleNewFile(file) {
  const validationError = validateFile(file);
  if (validationError) {
    selectedFile = null;
    imageInput.value = "";
    fileMeta.textContent = "No file selected yet.";
    resetPreview();
    outputText.value = "";
    setFeedback(validationError, "error");
    setStatus("Upload failed", "error");
    updateActions();
    return;
  }

  selectedFile = file;
  fileMeta.textContent = `${file.name} - ${formatFileSize(file.size)}`;
  setFeedback("Image ready for OCR.", "success");
  setStatus("Image loaded. Ready to run OCR.");
  previewSelectedFile(file);
  outputText.value = "";
  updateActions();
}

async function runOCR() {
  if (!selectedFile) {
    setFeedback("Select an image before running OCR.", "error");
    return;
  }

  setLoading(true);
  setFeedback("");
  setStatus("Sending image to OCR engine...");

  const formData = new FormData();
  formData.append("image", selectedFile);

  try {
    // The backend already exposes POST /ocr, so the frontend only needs to send the file as multipart form data.
    const response = await fetch(API_URL, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      let message = `OCR request failed with status ${response.status}.`;
      try {
        const errorData = await response.json();
        if (errorData?.error) {
          message = errorData.error;
        }
      } catch (parseError) {
        // Keep the fallback message when the error response is not JSON.
      }
      throw new Error(message);
    }

    const data = await response.json();
    const text = typeof data.text === "string" ? data.text.trim() : "";
    outputText.value = text;

    // OCR may succeed with an empty string, so treat that as a soft failure for the UI.
    if (text) {
      setFeedback("OCR completed successfully.", "success");
      setStatus("Text extracted successfully.", "success");
    } else {
      setFeedback("OCR completed, but no text was returned.", "error");
      setStatus("No text detected.", "error");
    }
  } catch (error) {
    outputText.value = "";
    setFeedback(error.message || "Unable to reach the OCR API.", "error");
    setStatus("OCR request failed.", "error");
  } finally {
    setLoading(false);
    updateActions();
  }
}

async function copyOutput() {
  const text = outputText.value.trim();
  if (!text) {
    return;
  }

  try {
    await navigator.clipboard.writeText(text);
    setFeedback("Recognized text copied to clipboard.", "success");
  } catch (error) {
    setFeedback("Copy failed. Your browser blocked clipboard access.", "error");
  }
}

function loadTheme() {
  const savedTheme = localStorage.getItem("sanskrit-ocr-theme");
  if (savedTheme === "dark") {
    document.body.classList.add("dark");
  }
}

function toggleTheme() {
  // Persist theme choice so the OCR workspace feels consistent across visits.
  document.body.classList.toggle("dark");
  localStorage.setItem(
    "sanskrit-ocr-theme",
    document.body.classList.contains("dark") ? "dark" : "light",
  );
}

imageInput.addEventListener("change", (event) => {
  const [file] = event.target.files || [];
  handleNewFile(file);
});

dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("is-dragging");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("is-dragging");
});

dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("is-dragging");
  const [file] = event.dataTransfer?.files || [];
  handleNewFile(file);
});

runButton.addEventListener("click", runOCR);
clearButton.addEventListener("click", resetForm);
copyButton.addEventListener("click", copyOutput);
themeToggle.addEventListener("click", toggleTheme);

loadTheme();
resetForm();
