(function () {
    const formData = new FormData();
    setUpDragAndDrop();
    setUpUploadButton();

    function setUpDragAndDrop() {
        const target = document.getElementById('target');

        target.addEventListener('drop', (e) => {
            e.stopPropagation();
            e.preventDefault();

            displayImage(e.dataTransfer.files);
        });

        target.addEventListener('dragover', (e) => {
            e.stopPropagation();
            e.preventDefault();

            e.dataTransfer.dropEffect = 'copy';
        });
    }

    function displayImage(fileList) {
        const output = document.getElementById('output');
        let file = null;

        for (let i = 0; i < fileList.length; i++) {
            if (fileList[i].type.match(/^image\//)) {
                file = fileList[i];
                break;
            }
        }

        if (file !== null) {
            output.src = URL.createObjectURL(file);
            formData.append('image', file);
            output.classList.add('output');
            const upload = document.getElementById('upload');
            upload.style.display = 'inline-block';
        }
    }

    function setUpUploadButton() {
        const upload = document.getElementById('upload');

        upload.addEventListener('click', () => {
            axios.post('/upload', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data'
                }
            });
        });
    }
})();