// Food Recognition App - JavaScript

document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('fileInput');
    const chooseFileBtn = document.getElementById('chooseFileBtn');
    const uploadArea = document.getElementById('uploadArea');
    const imagePreview = document.getElementById('imagePreview');
    const previewImg = document.getElementById('previewImg');
    const removeImageBtn = document.getElementById('removeImageBtn');
    const uploadForm = document.getElementById('uploadForm');
    const loader = document.getElementById('loader');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const portionSizeSlider = document.getElementById('portionSize');
    const portionSizeValue = document.getElementById('portionSizeValue');

    // Update portion size display
    if (portionSizeSlider && portionSizeValue) {
        portionSizeSlider.addEventListener('input', function() {
            portionSizeValue.textContent = this.value + 'g';
        });
    }

    // Click on upload area or button to trigger file input
    if (chooseFileBtn) {
        chooseFileBtn.addEventListener('click', function(e) {
            e.preventDefault();
            fileInput.click();
        });
    }

    if (uploadArea) {
        uploadArea.addEventListener('click', function(e) {
            if (e.target === uploadArea || e.target.closest('.upload-area') === uploadArea || e.target.closest('.upload-area-dark') === uploadArea) {
                fileInput.click();
            }
        });
    }

    // File input change handler
    if (fileInput) {
        fileInput.addEventListener('change', function(e) {
            handleFileSelect(e.target.files[0]);
        });
    }

    // Drag and drop handlers
    if (uploadArea) {
        uploadArea.addEventListener('dragover', function(e) {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', function(e) {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', function(e) {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                const file = files[0];
                if (file.type.startsWith('image/')) {
                    fileInput.files = files;
                    handleFileSelect(file);
                } else {
                    alert('Please upload an image file.');
                }
            }
        });
    }

    // Handle file selection
    function handleFileSelect(file) {
        if (!file) return;

        // Validate file type
        if (!file.type.startsWith('image/')) {
            alert('Please select an image file.');
            return;
        }

        // Validate file size (16MB)
        if (file.size > 16 * 1024 * 1024) {
            alert('File size must be less than 16MB.');
            return;
        }

        // Read and display preview
        const reader = new FileReader();
        reader.onload = function(e) {
            previewImg.src = e.target.result;
            imagePreview.classList.remove('d-none');
            uploadArea.querySelector('.text-center').style.display = 'none';
        };
        reader.readAsDataURL(file);
    }

    // Remove image
    if (removeImageBtn) {
        removeImageBtn.addEventListener('click', function() {
            fileInput.value = '';
            imagePreview.classList.add('d-none');
            uploadArea.querySelector('.text-center').style.display = 'block';
        });
    }

    // Form submission
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            // Check if file is selected
            if (!fileInput.files || fileInput.files.length === 0) {
                e.preventDefault();
                alert('Please select an image file first.');
                return;
            }

            // Show loader and hide button
            if (loader) {
                loader.classList.remove('d-none');
            }
            if (analyzeBtn) {
                analyzeBtn.disabled = true;
                analyzeBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status"></span>Analyzing...';
            }
        });
    }

    // Image load error handler for result page
    const resultImage = document.getElementById('resultImage');
    if (resultImage) {
        resultImage.addEventListener('error', function() {
            this.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="400" height="400"%3E%3Crect width="400" height="400" fill="%23f0f0f0"/%3E%3Ctext x="50%25" y="50%25" text-anchor="middle" dy=".3em" fill="%23999"%3EImage not found%3C/text%3E%3C/svg%3E';
        });
    }
});

// Add smooth scroll behavior
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

