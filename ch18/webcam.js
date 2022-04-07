// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

/**
 * 4D 텐서를 캡쳐하기 위해 웹캠 비디오 요소를 감싸는 클래스
 */

class Webcam {
    /**
    * @param {HTMLVideoElement} webcamElement 웹캠의 HTMLVideoElement
    */
    constructor(webcamElement) {
        this.webcamElement = webcamElement;
    }

    /**
    * 웹캠에서 이미지를 캡쳐하고 -1~1 사이로 정규화합니다.
    * [1, w, h, c] 크기의 (원소가 하나인) 배치 이미지를 반환합니다.
    */
    capture() {
        return tf.tidy(() => {
            // <video> 요소에서 이미지를 텐서로 읽습니다.
            const webcamImage = tf.browser.fromPixels(this.webcamElement);

            const reversedImage = webcamImage.reverse(1);

            // 이미지에서 중앙 부위의 정사각형을 잘라냅니다.
            const croppedImage = this.cropImage(reversedImage);

            // 배치 크기 1로 만들기 위해 첫 번째 차원을 추가합니다.
            const batchedImage = croppedImage.expandDims(0);

            // 이미지를 -1과 1 사이로 정규화합니다.
            // 이미지 픽셀 값이 0-255이므로 127로 나누고 1을 뺍니다.
            return batchedImage.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));
        });
    }

    /**
    * 공백이 없는 정사각형 이미지로 잘라냅니다.
    * @param {Tensor4D} img 이미지 텐서
    */
    cropImage(img) {
        const size = Math.min(img.shape[0], img.shape[1]);
        const centerHeight = img.shape[0] / 2;
        const beginHeight = centerHeight - (size / 2);
        const centerWidth = img.shape[1] / 2;
        const beginWidth = centerWidth - (size / 2);
        return img.slice([beginHeight, beginWidth, 0], [size, size, 3]);
    }

    /**
    * 공백 없이 크롭할 수 있도록 비디오 크기를 조정합니다.
    * @param {number} width 비디오 요소의 실제 너비
    * @param {number} height 비디오 요소의 실제 높이
    */
    adjustVideoSize(width, height) {
        const aspectRatio = width / height;
        if (width >= height) {
            this.webcamElement.width = aspectRatio * this.webcamElement.height;
        } else if (width < height) {
            this.webcamElement.height = this.webcamElement.width / aspectRatio;
        }
    }

    async setup() {
        return new Promise((resolve, reject) => {
            navigator.getUserMedia = navigator.getUserMedia ||
                navigator.webkitGetUserMedia || navigator.mozGetUserMedia ||
                navigator.msGetUserMedia;
            if (navigator.getUserMedia) {
                navigator.getUserMedia(
                    {video: {width: 224, height: 224}},
                    stream => {
                        this.webcamElement.srcObject = stream;
                        this.webcamElement.addEventListener('loadeddata', async () => {
                            this.adjustVideoSize(
                                this.webcamElement.videoWidth,
                                this.webcamElement.videoHeight);
                            resolve();
                        }, false);
                    },
                    error => {
                        reject(error);
                    }
                );
            } else {
                reject();
            }
        });
    }
}
