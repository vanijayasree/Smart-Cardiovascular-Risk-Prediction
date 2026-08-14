// Show alert if prediction is high risk
function showRiskAlert(message) {
    const alertBox = document.createElement("div");
    alertBox.className = "alert alert-danger alert-dismissible fade show";
    alertBox.role = "alert";
    alertBox.innerHTML = `
        <strong>⚠️ High Risk Detected!</strong> ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    document.querySelector("main").prepend(alertBox);
}

// Example: call when prediction = "High Risk"
function checkPrediction(prediction) {
    if (prediction === "High Risk") {
        showRiskAlert("Please consult a doctor immediately.");
    }
}
