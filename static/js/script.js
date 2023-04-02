function checkImage() {
  var fileInput = document.getElementById('file');
  var filePath = fileInput.value;
  var allowedExtensions = /(\.jpg|\.jpeg|\.png)$/i;
  var errorDiv = document.getElementById('error-msg');
  if (!filePath) {
    var errorMsg = 'Please upload a file.';
  } else if (!allowedExtensions.exec(filePath)) {
    var errorMsg = 'Please upload an image file (jpg, jpeg, or png).';
  }
  if (errorMsg) {
    if (!errorDiv) {
      errorDiv = document.createElement('div');
      errorDiv.id = 'error-msg';
      errorDiv.classList.add('active');
      var classifyButton = document.getElementById('classify-btn');
      classifyButton.parentNode.insertBefore(errorDiv, classifyButton.nextSibling);
    }
    errorDiv.innerHTML = errorMsg;
    errorDiv.classList.add('active');
    return false;
  }
  return true;
}
