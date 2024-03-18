// const video = document.getElementById('video');
// const recordButton = document.getElementById('record');
// const stopRecordButton = document.getElementById('stopRecord');
// const canvas = document.getElementById('canvas');
// const streamUrl = '/video_feed';

// let stream;
// let mediaRecorder;
// let chunks = [];

// async function startStream() {
//     stream = await navigator.mediaDevices.getUserMedia({ video: true });
//     video.srcObject = stream;
//     recordButton.disabled = false;
// }

// function startRecording() {
//     mediaRecorder = new MediaRecorder(stream);
//     mediaRecorder.ondataavailable = (event) => {
//         if (event.data.size > 0) {
//             chunks.push(event.data);
//         }
//     };
//     mediaRecorder.start();
//     recordButton.disabled = true;
//     stopRecordButton.disabled = false;
// }

// function stopRecording() {
//     mediaRecorder.stop();
//     recordButton.disabled = false;
//     stopRecordButton.disabled = true;
//     const blob = new Blob(chunks, { 'type' : 'video/webm' });
//     chunks = [];
//     const url = window.URL.createObjectURL(blob);
//     const a = document.createElement('a');
//     a.href = url;
//     a.download = 'recorded.webm';
//     document.body.appendChild(a);
//     a.click();
//     window.URL.revokeObjectURL(url);
// }

// startStream();
// recordButton.addEventListener('click', startRecording);
// stopRecordButton.addEventListener('click', stopRecording);

//streaming and cut video
// let recording = false;
// recordButton.addEventListener('click', () => {
//     if (!recording) {

//         recording = true;

  
//         recordButton.style.backgroundColor = '#7fdbff';
//         stopRecordButton.style.backgroundColor = '#ff4136';

//         recordButton.textContent = 'Recording...';

  
//     } else {
   
//         recording = false;

 
//         recordButton.style.backgroundColor = '#0074d9';
//         stopRecordButton.style.backgroundColor = '#ff4136';


//         recordButton.textContent = 'Start Record';

  
//     }
// });

// stopRecordButton.addEventListener('click', () => {

//     stopRecordButton.style.backgroundColor = '#ff4136';

// });

//print output in realtime
// function uploadFile() {
//     var formData = new FormData();
//     var fileInput = document.getElementById('file');
//     formData.append('file', fileInput.files[0]);

//     var xhr = new XMLHttpRequest();

//     xhr.open('POST', '/upload', true);

//     xhr.onreadystatechange = function() {
//         if (xhr.readyState === 3 && xhr.status === 200) {
//             var responseText = xhr.responseText;
           
//             document.getElementById('result').value = responseText;
//         } else if (xhr.readyState === 4 && xhr.status !== 200) {
//             document.getElementById('result').innerHTML = "Có lỗi xảy ra khi tải lên tệp.";
//         }
//     };

//     xhr.send(formData);
// }

function pollForResults() {
    setInterval(function() {
        var xhr = new XMLHttpRequest();
        xhr.open('GET', '/get_latest_result', true);

        xhr.onreadystatechange = function() {
            if (xhr.readyState === 4 && xhr.status === 200) {
                var resultTextarea = document.getElementById('result-textarea');
                var responseData = JSON.parse(xhr.responseText);
                var result = responseData.result;
                resultTextarea.value = result;
            }
        };

        xhr.send();
    }, 2000);  //update result every 2 seconds
}

pollForResults();

//clear result
document.getElementById('clear-button').addEventListener('click', function() {
    fetch('/clear_all_res', {method: 'POST'})
    .then(response => response.json())
    .then(data => {
        if (data.result === 'success') {
            document.getElementById('result-textarea').value = '';
        }
    });
});

//import video
var uploadPath = window.location.origin + '/uploads/';
document.getElementById('upload-button').addEventListener('click', function() {
    var file = document.getElementById('file-input').files[0];
    var formData = new FormData();
    formData.append('file', file);
    fetch('/upload', { method: 'POST', body: formData })
        .then(response => response.json())
        .then(data => {
            if (data.result === 'success') {
                // The file was successfully uploaded and you can use the filename here
                var filename = data.filename;
                // Change the source of the video element to play the uploaded video
                var userUploadedVideo = document.getElementById('user-uploaded-video');
                userUploadedVideo.src = uploadPath + filename;
                userUploadedVideo.style.display = 'block'; 

                
                var videoFeed = document.getElementById('video-img');
                videoFeed.style.display = 'none'; 
            }
        });
});


