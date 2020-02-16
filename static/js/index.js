(function () {
    const target = document.getElementById('target');

    target.addEventListener('drop', (e) => {
        e.stopPropagation();
        e.preventDefault();

        console.log(e.dataTransfer.files);
    });

    target.addEventListener('dragover', (e) => {
        e.stopPropagation();
        e.preventDefault();

        e.dataTransfer.dropEffect = 'copy';
    });
})();